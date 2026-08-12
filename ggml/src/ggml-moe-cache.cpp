#include "ggml-moe-cache.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>

enum class ggml_moe_cache_segment {
    probation,
    protected_lru,
};

struct ggml_moe_cache_entry {
    size_t size_bytes;
    uint32_t group;
    ggml_moe_cache_segment segment;
    std::list<uint64_t>::iterator position;
};

struct ggml_moe_cache_policy::impl {
    impl(
            size_t capacity_bytes,
            uint32_t protected_percent,
            uint32_t admission_threshold,
            uint64_t frequency_decay_interval) :
        capacity(capacity_bytes),
        protected_target(capacity_bytes / 100 * protected_percent + capacity_bytes % 100 * protected_percent / 100),
        admission_threshold(std::max(1u, admission_threshold)),
        frequency_decay_interval(frequency_decay_interval) {
        assert(protected_percent <= 100);
    }

    size_t capacity;
    size_t protected_target;
    uint32_t admission_threshold;
    uint64_t frequency_decay_interval;
    uint64_t clock = 0;
    size_t resident = 0;
    size_t protected_size = 0;
    size_t probation_size = 0;
    uint32_t group_count = 0;
    std::vector<size_t> group_resident;
    ggml_moe_cache_stats stats;
    std::list<uint64_t> probation;
    std::list<uint64_t> protected_lru;
    std::unordered_map<uint64_t, ggml_moe_cache_entry> entries;
    std::unordered_map<uint64_t, uint32_t> frequencies;

    void decay_frequencies() {
        if (frequency_decay_interval == 0 || clock % frequency_decay_interval != 0) {
            return;
        }

        for (auto it = frequencies.begin(); it != frequencies.end();) {
            it->second /= 2;
            if (it->second == 0 && entries.find(it->first) == entries.end()) {
                it = frequencies.erase(it);
            } else {
                ++it;
            }
        }
    }

    uint32_t record_frequency(uint64_t key) {
        uint32_t & value = frequencies[key];
        if (value != std::numeric_limits<uint32_t>::max()) {
            ++value;
        }
        return value;
    }

    void move_to_front(ggml_moe_cache_entry & entry, uint64_t key) {
        std::list<uint64_t> & lru = entry.segment == ggml_moe_cache_segment::protected_lru ? protected_lru : probation;
        lru.erase(entry.position);
        lru.push_front(key);
        entry.position = lru.begin();
    }

    void demote_protected_lru() {
        assert(!protected_lru.empty());
        const uint64_t key = protected_lru.back();
        protected_lru.pop_back();

        auto entry_it = entries.find(key);
        assert(entry_it != entries.end());
        ggml_moe_cache_entry & entry = entry_it->second;
        protected_size -= entry.size_bytes;
        probation_size += entry.size_bytes;
        entry.segment = ggml_moe_cache_segment::probation;
        probation.push_front(key);
        entry.position = probation.begin();
    }

    void balance_protected() {
        while (protected_size > protected_target && !protected_lru.empty()) {
            demote_protected_lru();
        }
    }

    void remove_group_resident(const ggml_moe_cache_entry & entry) {
        if (entry.group < group_resident.size()) {
            assert(group_resident[entry.group] >= entry.size_bytes);
            group_resident[entry.group] -= entry.size_bytes;
        }
    }

    uint32_t most_over_budget_group() const {
        if (group_count == 0) {
            return UINT32_MAX;
        }

        const size_t target = capacity / group_count;
        uint32_t selected = UINT32_MAX;
        size_t selected_excess = 0;
        for (uint32_t group = 0; group < group_count; ++group) {
            const size_t bytes = group_resident[group];
            if (bytes > target && bytes - target > selected_excess) {
                selected = group;
                selected_excess = bytes - target;
            }
        }
        return selected;
    }

    std::list<uint64_t>::iterator find_lru(std::list<uint64_t> & lru, uint32_t group) {
        if (group == UINT32_MAX) {
            return lru.empty() ? lru.end() : std::prev(lru.end());
        }
        for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
            const auto entry_it = entries.find(*it);
            assert(entry_it != entries.end());
            if (entry_it->second.group == group) {
                return std::prev(it.base());
            }
        }
        return lru.end();
    }

    void demote_protected(std::list<uint64_t>::iterator position) {
        const uint64_t key = *position;
        protected_lru.erase(position);

        auto entry_it = entries.find(key);
        assert(entry_it != entries.end());
        ggml_moe_cache_entry & entry = entry_it->second;
        protected_size -= entry.size_bytes;
        probation_size += entry.size_bytes;
        entry.segment = ggml_moe_cache_segment::probation;
        probation.push_front(key);
        entry.position = probation.begin();
    }

    uint64_t evict_probation(std::list<uint64_t>::iterator position) {
        assert(position != probation.end());
        const uint64_t key = *position;
        probation.erase(position);

        auto entry_it = entries.find(key);
        assert(entry_it != entries.end());
        const size_t size_bytes = entry_it->second.size_bytes;
        remove_group_resident(entry_it->second);
        probation_size -= size_bytes;
        resident -= size_bytes;
        entries.erase(entry_it);
        ++stats.evictions;
        return key;
    }

    bool evict_one(uint64_t & key) {
        const uint32_t group = most_over_budget_group();
        auto probation_it = find_lru(probation, group);
        if (probation_it == probation.end()) {
            auto protected_it = find_lru(protected_lru, group);
            if (protected_it == protected_lru.end()) {
                probation_it = find_lru(probation, UINT32_MAX);
                if (probation_it == probation.end()) {
                    protected_it = find_lru(protected_lru, UINT32_MAX);
                }
            }
            if (protected_it != protected_lru.end()) {
                demote_protected(protected_it);
                probation_it = probation.begin();
            }
        }
        if (probation_it == probation.end()) {
            return false;
        }
        key = evict_probation(probation_it);
        return true;
    }

    void make_room(size_t size_bytes, std::vector<uint64_t> & evicted) {
        while (size_bytes > capacity - resident) {
            uint64_t key;
            const bool evicted_one = evict_one(key);
            assert(evicted_one);
            if (!evicted_one) {
                break;
            }
            evicted.push_back(key);
        }
    }

    void promote(uint64_t key, ggml_moe_cache_entry & entry) {
        assert(entry.segment == ggml_moe_cache_segment::probation);
        probation.erase(entry.position);
        probation_size -= entry.size_bytes;
        protected_size += entry.size_bytes;
        entry.segment = ggml_moe_cache_segment::protected_lru;
        protected_lru.push_front(key);
        entry.position = protected_lru.begin();
        balance_protected();
    }
};

ggml_moe_cache_policy::ggml_moe_cache_policy(
        size_t capacity_bytes,
        uint32_t protected_percent,
        uint32_t admission_threshold,
        uint64_t frequency_decay_interval) :
    pimpl(new impl(capacity_bytes, protected_percent, admission_threshold, frequency_decay_interval)) {
}

ggml_moe_cache_policy::~ggml_moe_cache_policy() = default;

ggml_moe_cache_result ggml_moe_cache_policy::access(uint64_t key, size_t size_bytes, uint32_t group) {
    ggml_moe_cache_result result;
    ++pimpl->clock;
    ++pimpl->stats.accesses;
    pimpl->decay_frequencies();
    const uint32_t frequency = pimpl->record_frequency(key);

    auto entry_it = pimpl->entries.find(key);
    if (entry_it != pimpl->entries.end()) {
        assert(entry_it->second.size_bytes == size_bytes);
        assert(entry_it->second.group == group);
        result.hit = true;
        ++pimpl->stats.hits;
        if (entry_it->second.segment == ggml_moe_cache_segment::probation) {
            pimpl->promote(key, entry_it->second);
        } else {
            pimpl->move_to_front(entry_it->second, key);
        }
        return result;
    }

    ++pimpl->stats.misses;
    if (size_bytes == 0 || size_bytes > pimpl->capacity || frequency < pimpl->admission_threshold) {
        return result;
    }

    pimpl->make_room(size_bytes, result.evicted);
    pimpl->probation.push_front(key);
    pimpl->entries.emplace(key, ggml_moe_cache_entry {
        size_bytes,
        group,
        ggml_moe_cache_segment::probation,
        pimpl->probation.begin(),
    });
    pimpl->probation_size += size_bytes;
    pimpl->resident += size_bytes;
    if (group < pimpl->group_resident.size()) {
        pimpl->group_resident[group] += size_bytes;
    }
    ++pimpl->stats.admission_attempts;
    ++pimpl->stats.admissions;
    result.admitted = true;
    return result;
}

bool ggml_moe_cache_policy::erase(uint64_t key) {
    auto entry_it = pimpl->entries.find(key);
    if (entry_it == pimpl->entries.end()) {
        return false;
    }

    const ggml_moe_cache_entry & entry = entry_it->second;
    if (entry.segment == ggml_moe_cache_segment::protected_lru) {
        pimpl->protected_lru.erase(entry.position);
        pimpl->protected_size -= entry.size_bytes;
    } else {
        pimpl->probation.erase(entry.position);
        pimpl->probation_size -= entry.size_bytes;
    }
    pimpl->remove_group_resident(entry);
    pimpl->resident -= entry.size_bytes;
    pimpl->entries.erase(entry_it);
    return true;
}

bool ggml_moe_cache_policy::evict(uint64_t & key) {
    return pimpl->evict_one(key);
}

void ggml_moe_cache_policy::set_group_count(uint32_t group_count) {
    if (group_count <= pimpl->group_count) {
        return;
    }
    pimpl->group_count = group_count;
    pimpl->group_resident.resize(group_count, 0);
    std::fill(pimpl->group_resident.begin(), pimpl->group_resident.end(), 0);
    for (const auto & item : pimpl->entries) {
        const auto & entry = item.second;
        if (entry.group < pimpl->group_resident.size()) {
            pimpl->group_resident[entry.group] += entry.size_bytes;
        }
    }
}

bool ggml_moe_cache_policy::contains(uint64_t key) const {
    return pimpl->entries.find(key) != pimpl->entries.end();
}

size_t ggml_moe_cache_policy::capacity_bytes() const {
    return pimpl->capacity;
}

size_t ggml_moe_cache_policy::resident_bytes() const {
    return pimpl->resident;
}

size_t ggml_moe_cache_policy::protected_bytes() const {
    return pimpl->protected_size;
}

size_t ggml_moe_cache_policy::probation_bytes() const {
    return pimpl->probation_size;
}

size_t ggml_moe_cache_policy::group_resident_bytes(uint32_t group) const {
    return group < pimpl->group_resident.size() ? pimpl->group_resident[group] : 0;
}

size_t ggml_moe_cache_policy::size() const {
    return pimpl->entries.size();
}

uint32_t ggml_moe_cache_policy::frequency(uint64_t key) const {
    const auto it = pimpl->frequencies.find(key);
    return it == pimpl->frequencies.end() ? 0 : it->second;
}

const ggml_moe_cache_stats & ggml_moe_cache_policy::stats() const {
    return pimpl->stats;
}

struct ggml_moe_cache_slot {
    size_t offset;
    size_t size_bytes;
};

struct ggml_moe_cache_slots::impl {
    impl(
            size_t capacity_bytes,
            size_t alignment,
            uint32_t protected_percent,
            uint32_t admission_threshold,
            uint64_t frequency_decay_interval) :
        capacity(capacity_bytes),
        alignment(std::max((size_t) 1, alignment)),
        policy(capacity_bytes, protected_percent, admission_threshold, frequency_decay_interval) {
        if (capacity > 0) {
            free_ranges.emplace(0, capacity);
        }
    }

    size_t capacity;
    size_t alignment;
    ggml_moe_cache_policy policy;
    std::map<size_t, size_t> free_ranges;
    std::unordered_map<uint64_t, ggml_moe_cache_slot> slots;
    ggml_moe_cache_stats stats;

    size_t aligned_size(size_t size_bytes) const {
        if (size_bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
            return std::numeric_limits<size_t>::max();
        }
        return (size_bytes + alignment - 1) / alignment * alignment;
    }

    size_t allocate(size_t size_bytes) {
        for (auto it = free_ranges.begin(); it != free_ranges.end(); ++it) {
            if (it->second < size_bytes) {
                continue;
            }

            const size_t offset = it->first;
            const size_t remaining = it->second - size_bytes;
            free_ranges.erase(it);
            if (remaining > 0) {
                free_ranges.emplace(offset + size_bytes, remaining);
            }
            return offset;
        }
        return SIZE_MAX;
    }

    void release(size_t offset, size_t size_bytes) {
        auto next = free_ranges.lower_bound(offset);
        if (next != free_ranges.begin()) {
            auto previous = std::prev(next);
            if (previous->first + previous->second == offset) {
                offset = previous->first;
                size_bytes += previous->second;
                free_ranges.erase(previous);
            }
        }
        if (next != free_ranges.end() && offset + size_bytes == next->first) {
            size_bytes += next->second;
            free_ranges.erase(next);
        }
        free_ranges.emplace(offset, size_bytes);
    }

    void release_key(uint64_t key) {
        const auto it = slots.find(key);
        assert(it != slots.end());
        release(it->second.offset, it->second.size_bytes);
        slots.erase(it);
    }
};

ggml_moe_cache_slots::ggml_moe_cache_slots(
        size_t capacity_bytes,
        size_t alignment,
        uint32_t protected_percent,
        uint32_t admission_threshold,
        uint64_t frequency_decay_interval) :
    pimpl(new impl(capacity_bytes, alignment, protected_percent, admission_threshold, frequency_decay_interval)) {
}

ggml_moe_cache_slots::~ggml_moe_cache_slots() = default;

ggml_moe_cache_slot_result ggml_moe_cache_slots::access(uint64_t key, size_t size_bytes, uint32_t group) {
    ggml_moe_cache_slot_result result;
    const size_t allocation_size = pimpl->aligned_size(size_bytes);
    auto policy_result = pimpl->policy.access(key, allocation_size, group);
    result.hit = policy_result.hit;
    result.evicted = std::move(policy_result.evicted);
    ++pimpl->stats.accesses;

    if (result.hit) {
        ++pimpl->stats.hits;
        const auto it = pimpl->slots.find(key);
        assert(it != pimpl->slots.end());
        result.offset = it->second.offset;
        return result;
    }

    ++pimpl->stats.misses;

    if (!policy_result.admitted) {
        return result;
    }
    ++pimpl->stats.admission_attempts;

    for (uint64_t evicted : result.evicted) {
        pimpl->release_key(evicted);
    }

    size_t offset = pimpl->allocate(allocation_size);
    if (offset == SIZE_MAX) {
        const bool erased = pimpl->policy.erase(key);
        assert(erased);
        if (!erased) {
            ++pimpl->stats.allocation_failures;
            return result;
        }
        while (offset == SIZE_MAX) {
            uint64_t evicted;
            if (!pimpl->policy.evict(evicted)) {
                break;
            }
            pimpl->release_key(evicted);
            result.evicted.push_back(evicted);
            offset = pimpl->allocate(allocation_size);
        }
        if (offset != SIZE_MAX) {
            const auto restored = pimpl->policy.access(key, allocation_size, group);
            assert(restored.admitted && restored.evicted.empty());
        }
    }
    pimpl->stats.evictions += result.evicted.size();
    if (offset == SIZE_MAX) {
        pimpl->policy.erase(key);
        ++pimpl->stats.allocation_failures;
        return result;
    }

    pimpl->slots.emplace(key, ggml_moe_cache_slot { offset, allocation_size });
    ++pimpl->stats.admissions;
    result.admitted = true;
    result.offset = offset;
    return result;
}

void ggml_moe_cache_slots::set_group_count(uint32_t group_count) {
    pimpl->policy.set_group_count(group_count);
}

void ggml_moe_cache_slots::reset_stats() {
    pimpl->stats = {};
}

bool ggml_moe_cache_slots::erase(uint64_t key) {
    if (!pimpl->policy.erase(key)) {
        return false;
    }
    pimpl->release_key(key);
    return true;
}

bool ggml_moe_cache_slots::contains(uint64_t key) const {
    return pimpl->policy.contains(key);
}

size_t ggml_moe_cache_slots::offset(uint64_t key) const {
    const auto it = pimpl->slots.find(key);
    return it == pimpl->slots.end() ? SIZE_MAX : it->second.offset;
}

size_t ggml_moe_cache_slots::resident_bytes() const {
    return pimpl->policy.resident_bytes();
}

size_t ggml_moe_cache_slots::resident_entries() const {
    return pimpl->slots.size();
}

size_t ggml_moe_cache_slots::largest_free_range() const {
    size_t largest = 0;
    for (const auto & range : pimpl->free_ranges) {
        largest = std::max(largest, range.second);
    }
    return largest;
}

size_t ggml_moe_cache_slots::group_resident_bytes(uint32_t group) const {
    return pimpl->policy.group_resident_bytes(group);
}

const ggml_moe_cache_stats & ggml_moe_cache_slots::stats() const {
    return pimpl->stats;
}
