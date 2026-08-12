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

    uint64_t evict_probation_lru() {
        assert(!probation.empty());
        const uint64_t key = probation.back();
        probation.pop_back();

        auto entry_it = entries.find(key);
        assert(entry_it != entries.end());
        const size_t size_bytes = entry_it->second.size_bytes;
        probation_size -= size_bytes;
        resident -= size_bytes;
        entries.erase(entry_it);
        ++stats.evictions;
        return key;
    }

    void make_room(size_t size_bytes, std::vector<uint64_t> & evicted) {
        while (size_bytes > capacity - resident) {
            if (probation.empty()) {
                assert(!protected_lru.empty());
                demote_protected_lru();
            }
            evicted.push_back(evict_probation_lru());
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

ggml_moe_cache_result ggml_moe_cache_policy::access(uint64_t key, size_t size_bytes) {
    ggml_moe_cache_result result;
    ++pimpl->clock;
    ++pimpl->stats.accesses;
    pimpl->decay_frequencies();
    const uint32_t frequency = pimpl->record_frequency(key);

    auto entry_it = pimpl->entries.find(key);
    if (entry_it != pimpl->entries.end()) {
        assert(entry_it->second.size_bytes == size_bytes);
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
        ggml_moe_cache_segment::probation,
        pimpl->probation.begin(),
    });
    pimpl->probation_size += size_bytes;
    pimpl->resident += size_bytes;
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
    pimpl->resident -= entry.size_bytes;
    pimpl->entries.erase(entry_it);
    return true;
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

ggml_moe_cache_slot_result ggml_moe_cache_slots::access(uint64_t key, size_t size_bytes) {
    ggml_moe_cache_slot_result result;
    const size_t allocation_size = pimpl->aligned_size(size_bytes);
    auto policy_result = pimpl->policy.access(key, allocation_size);
    result.hit = policy_result.hit;
    result.evicted = std::move(policy_result.evicted);

    if (result.hit) {
        const auto it = pimpl->slots.find(key);
        assert(it != pimpl->slots.end());
        result.offset = it->second.offset;
        return result;
    }

    if (!policy_result.admitted) {
        return result;
    }

    for (uint64_t evicted : result.evicted) {
        pimpl->release_key(evicted);
    }

    const size_t offset = pimpl->allocate(allocation_size);
    if (offset == SIZE_MAX) {
        pimpl->policy.erase(key);
        return result;
    }

    pimpl->slots.emplace(key, ggml_moe_cache_slot { offset, allocation_size });
    result.admitted = true;
    result.offset = offset;
    return result;
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

const ggml_moe_cache_stats & ggml_moe_cache_slots::stats() const {
    return pimpl->policy.stats();
}
