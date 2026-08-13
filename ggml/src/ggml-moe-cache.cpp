#include "ggml-moe-cache.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
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

struct ggml_moe_cache_history {
    uint32_t short_frequency = 0;
    uint32_t long_frequency = 0;
};

std::vector<size_t> ggml_moe_layer_budgets(
        size_t capacity_bytes,
        uint32_t layer_count,
        const std::vector<ggml_moe_layer_candidate> & candidates) {
    std::vector<size_t> budgets(layer_count, 0);
    if (capacity_bytes == 0 || layer_count == 0) {
        return budgets;
    }

    std::vector<std::vector<const ggml_moe_layer_candidate *>> by_layer(layer_count);
    for (const auto & candidate : candidates) {
        if (candidate.layer < layer_count && candidate.size_bytes > 0 && candidate.size_bytes <= capacity_bytes) {
            by_layer[candidate.layer].push_back(&candidate);
        }
    }
    auto score_greater = [](const auto * a, const auto * b) {
        const long double a_score = (long double) a->expected_hits * a->read_bytes / a->size_bytes;
        const long double b_score = (long double) b->expected_hits * b->read_bytes / b->size_bytes;
        return a_score != b_score ? a_score > b_score : a->size_bytes < b->size_bytes;
    };
    for (auto & layer : by_layer) {
        std::sort(layer.begin(), layer.end(), score_greater);
    }

    size_t used = 0;
    std::vector<size_t> next(layer_count, 0);
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        if (by_layer[layer].empty()) {
            continue;
        }
        const size_t size_bytes = by_layer[layer][0]->size_bytes;
        if (size_bytes <= capacity_bytes - used) {
            budgets[layer] += size_bytes;
            used += size_bytes;
            next[layer] = 1;
        }
    }

    while (used < capacity_bytes) {
        uint32_t selected_layer = UINT32_MAX;
        for (uint32_t layer = 0; layer < layer_count; ++layer) {
            if (next[layer] >= by_layer[layer].size()) {
                continue;
            }
            const auto * candidate = by_layer[layer][next[layer]];
            if (candidate->size_bytes > capacity_bytes - used) {
                continue;
            }
            if (selected_layer == UINT32_MAX || score_greater(candidate, by_layer[selected_layer][next[selected_layer]])) {
                selected_layer = layer;
            }
        }
        if (selected_layer == UINT32_MAX) {
            break;
        }
        const size_t size_bytes = by_layer[selected_layer][next[selected_layer]]->size_bytes;
        budgets[selected_layer] += size_bytes;
        used += size_bytes;
        ++next[selected_layer];
    }

    uint32_t residual_layer = 0;
    for (uint32_t layer = 1; layer < layer_count; ++layer) {
        if (budgets[layer] > budgets[residual_layer]) {
            residual_layer = layer;
        }
    }
    budgets[residual_layer] += capacity_bytes - used;
    return budgets;
}

struct ggml_moe_cache_policy::impl {
    impl(
            size_t capacity_bytes,
            uint32_t protected_percent,
            uint32_t admission_threshold,
            uint64_t frequency_decay_interval,
            uint64_t short_frequency_decay_interval) :
        capacity(capacity_bytes),
        protected_target(capacity_bytes / 100 * protected_percent + capacity_bytes % 100 * protected_percent / 100),
        admission_threshold(std::max(1u, admission_threshold)),
        frequency_decay_interval(frequency_decay_interval),
        short_frequency_decay_interval(short_frequency_decay_interval) {
        assert(protected_percent <= 100);
    }

    size_t capacity;
    size_t protected_target;
    uint32_t admission_threshold;
    uint64_t frequency_decay_interval;
    uint64_t short_frequency_decay_interval;
    uint64_t clock = 0;
    size_t resident = 0;
    size_t protected_size = 0;
    size_t probation_size = 0;
    uint32_t group_count = 0;
    std::vector<size_t> group_resident;
    std::vector<size_t> group_budgets;
    bool custom_group_budgets = false;
    ggml_moe_cache_stats stats;
    std::list<uint64_t> probation;
    std::list<uint64_t> protected_lru;
    std::unordered_map<uint64_t, ggml_moe_cache_entry> entries;
    std::unordered_map<uint64_t, ggml_moe_cache_history> histories;

    void decay_frequencies() {
        const bool decay_short = short_frequency_decay_interval != 0 && clock % short_frequency_decay_interval == 0;
        const bool decay_long = frequency_decay_interval != 0 && clock % frequency_decay_interval == 0;
        if (!decay_short && !decay_long) {
            return;
        }

        for (auto it = histories.begin(); it != histories.end();) {
            if (decay_short) {
                it->second.short_frequency /= 2;
            }
            if (decay_long) {
                it->second.long_frequency /= 2;
            }
            if (it->second.short_frequency == 0 && it->second.long_frequency == 0 && entries.find(it->first) == entries.end()) {
                it = histories.erase(it);
            } else {
                ++it;
            }
        }
    }

    ggml_moe_cache_history record_frequency(uint64_t key) {
        ggml_moe_cache_history & history = histories[key];
        if (history.short_frequency != std::numeric_limits<uint32_t>::max()) {
            ++history.short_frequency;
        }
        if (history.long_frequency != std::numeric_limits<uint32_t>::max()) {
            ++history.long_frequency;
        }
        return history;
    }

    uint64_t benefit(uint64_t key) const {
        const auto it = histories.find(key);
        if (it == histories.end()) {
            return 0;
        }
        return (uint64_t) it->second.short_frequency * 4 + it->second.long_frequency;
    }

    uint64_t scaled_benefit(uint64_t key) const {
        return benefit(key) * 1024 / 5;
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

    uint32_t most_over_budget_group(const std::vector<size_t> & resident_by_group) const {
        if (group_count == 0) {
            return UINT32_MAX;
        }

        uint32_t selected = UINT32_MAX;
        size_t selected_excess = 0;
        for (uint32_t group = 0; group < group_count; ++group) {
            const size_t target = group < group_budgets.size() ? group_budgets[group] : capacity / group_count;
            const size_t bytes = resident_by_group[group];
            if (bytes > target && bytes - target > selected_excess) {
                selected = group;
                selected_excess = bytes - target;
            }
        }
        return selected;
    }

    uint32_t most_over_budget_group() const {
        return most_over_budget_group(group_resident);
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

    bool find_victim(
            uint64_t & key,
            const std::vector<size_t> & resident_by_group,
            const std::unordered_set<uint64_t> & excluded) const {
        auto find = [&](const std::list<uint64_t> & lru, uint32_t group) {
            for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
                if (excluded.find(*it) != excluded.end()) {
                    continue;
                }
                const auto entry_it = entries.find(*it);
                assert(entry_it != entries.end());
                if (group == UINT32_MAX || entry_it->second.group == group) {
                    key = *it;
                    return true;
                }
            }
            return false;
        };

        const uint32_t group = most_over_budget_group(resident_by_group);
        return find(probation, group) ||
               find(protected_lru, group) ||
               find(probation, UINT32_MAX) ||
               find(protected_lru, UINT32_MAX);
    }

    bool find_victims(size_t size_bytes, std::vector<uint64_t> & victims) const {
        size_t simulated_resident = resident;
        std::vector<size_t> simulated_groups = group_resident;
        std::unordered_set<uint64_t> excluded;
        while (size_bytes > capacity - simulated_resident) {
            uint64_t victim;
            if (!find_victim(victim, simulated_groups, excluded)) {
                return false;
            }
            const auto entry_it = entries.find(victim);
            assert(entry_it != entries.end());
            const ggml_moe_cache_entry & entry = entry_it->second;
            simulated_resident -= entry.size_bytes;
            if (entry.group < simulated_groups.size()) {
                simulated_groups[entry.group] -= entry.size_bytes;
            }
            victims.push_back(victim);
            excluded.insert(victim);
        }
        return true;
    }

    bool cost_allows(size_t size_bytes, uint64_t candidate_benefit) const {
        if (size_bytes <= capacity - resident) {
            return true;
        }
        std::vector<uint64_t> victims;
        if (!find_victims(size_bytes, victims)) {
            return false;
        }
        long double victim_value = 0;
        for (uint64_t victim : victims) {
            const auto victim_it = entries.find(victim);
            assert(victim_it != entries.end());
            victim_value += (long double) scaled_benefit(victim) * victim_it->second.size_bytes;
        }
        const long double candidate_value = (long double) candidate_benefit * size_bytes;
        return candidate_value * 4 > victim_value * 5;
    }

    ggml_moe_cache_result admit(uint64_t key, size_t size_bytes, uint32_t group) {
        ggml_moe_cache_result result;
        auto entry_it = entries.find(key);
        if (entry_it != entries.end()) {
            assert(entry_it->second.size_bytes == size_bytes);
            assert(entry_it->second.group == group);
            result.hit = true;
            return result;
        }
        if (size_bytes == 0 || size_bytes > capacity) {
            return result;
        }

        make_room(size_bytes, result.evicted);
        probation.push_front(key);
        entries.emplace(key, ggml_moe_cache_entry {
            size_bytes,
            group,
            ggml_moe_cache_segment::probation,
            probation.begin(),
        });
        probation_size += size_bytes;
        resident += size_bytes;
        if (group < group_resident.size()) {
            group_resident[group] += size_bytes;
        }
        ++stats.admission_attempts;
        ++stats.admissions;
        result.admitted = true;
        return result;
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
        uint64_t frequency_decay_interval,
        uint64_t short_frequency_decay_interval) :
    pimpl(new impl(capacity_bytes, protected_percent, admission_threshold, frequency_decay_interval, short_frequency_decay_interval)) {
}

ggml_moe_cache_policy::~ggml_moe_cache_policy() = default;

ggml_moe_cache_result ggml_moe_cache_policy::access(uint64_t key, size_t size_bytes, uint32_t group) {
    ggml_moe_cache_result result;
    ++pimpl->clock;
    ++pimpl->stats.accesses;
    pimpl->decay_frequencies();
    const auto history = pimpl->record_frequency(key);

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
    if (size_bytes == 0 || size_bytes > pimpl->capacity ||
            std::max(history.short_frequency, history.long_frequency) < pimpl->admission_threshold) {
        return result;
    }
    if (!pimpl->cost_allows(size_bytes, pimpl->scaled_benefit(key))) {
        ++pimpl->stats.cost_rejections;
        result.cost_rejected = true;
        return result;
    }

    result = pimpl->admit(key, size_bytes, group);
    return result;
}

ggml_moe_cache_result ggml_moe_cache_policy::prefetch(uint64_t key, size_t size_bytes, uint32_t group, uint64_t predicted_benefit) {
    if (pimpl->entries.find(key) != pimpl->entries.end()) {
        return pimpl->admit(key, size_bytes, group);
    }
    if (predicted_benefit != UINT64_MAX && !pimpl->cost_allows(size_bytes, predicted_benefit)) {
        ++pimpl->stats.cost_rejections;
        ggml_moe_cache_result result;
        result.cost_rejected = true;
        return result;
    }
    return pimpl->admit(key, size_bytes, group);
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
    if (!pimpl->custom_group_budgets) {
        pimpl->group_budgets.resize(group_count);
        for (uint32_t group = 0; group < group_count; ++group) {
            pimpl->group_budgets[group] = pimpl->capacity / group_count + (group < pimpl->capacity % group_count);
        }
    }
    std::fill(pimpl->group_resident.begin(), pimpl->group_resident.end(), 0);
    for (const auto & item : pimpl->entries) {
        const auto & entry = item.second;
        if (entry.group < pimpl->group_resident.size()) {
            pimpl->group_resident[entry.group] += entry.size_bytes;
        }
    }
}

void ggml_moe_cache_policy::set_group_budgets(const std::vector<size_t> & budgets) {
    if (budgets.empty()) {
        return;
    }
    set_group_count((uint32_t) budgets.size());
    pimpl->group_budgets = budgets;
    pimpl->custom_group_budgets = true;
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
    const auto it = pimpl->histories.find(key);
    return it == pimpl->histories.end() ? 0 : it->second.long_frequency;
}

uint32_t ggml_moe_cache_policy::short_frequency(uint64_t key) const {
    const auto it = pimpl->histories.find(key);
    return it == pimpl->histories.end() ? 0 : it->second.short_frequency;
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
            uint64_t frequency_decay_interval,
            uint64_t short_frequency_decay_interval) :
        capacity(capacity_bytes),
        alignment(std::max((size_t) 1, alignment)),
        policy(capacity_bytes, protected_percent, admission_threshold, frequency_decay_interval, short_frequency_decay_interval) {
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

    ggml_moe_cache_slot_result finalize(
            uint64_t key,
            size_t allocation_size,
            uint32_t group,
            ggml_moe_cache_result policy_result) {
        ggml_moe_cache_slot_result result;
        result.hit = policy_result.hit;
        result.cost_rejected = policy_result.cost_rejected;
        result.evicted = std::move(policy_result.evicted);
        if (result.hit) {
            result.offset = slots.at(key).offset;
            return result;
        }
        if (!policy_result.admitted) {
            return result;
        }
        ++stats.admission_attempts;
        for (uint64_t evicted : result.evicted) {
            release_key(evicted);
        }

        size_t offset = allocate(allocation_size);
        if (offset == SIZE_MAX) {
            const bool erased = policy.erase(key);
            assert(erased);
            if (!erased) {
                ++stats.allocation_failures;
                return result;
            }
            while (offset == SIZE_MAX) {
                uint64_t evicted;
                if (!policy.evict(evicted)) {
                    break;
                }
                release_key(evicted);
                result.evicted.push_back(evicted);
                offset = allocate(allocation_size);
            }
            if (offset != SIZE_MAX) {
                const auto restored = policy.prefetch(key, allocation_size, group);
                assert(restored.admitted && restored.evicted.empty());
            }
        }
        stats.evictions += result.evicted.size();
        if (offset == SIZE_MAX) {
            policy.erase(key);
            ++stats.allocation_failures;
            return result;
        }

        slots.emplace(key, ggml_moe_cache_slot { offset, allocation_size });
        ++stats.admissions;
        result.admitted = true;
        result.offset = offset;
        return result;
    }
};

ggml_moe_cache_slots::ggml_moe_cache_slots(
        size_t capacity_bytes,
        size_t alignment,
        uint32_t protected_percent,
        uint32_t admission_threshold,
        uint64_t frequency_decay_interval,
        uint64_t short_frequency_decay_interval) :
    pimpl(new impl(capacity_bytes, alignment, protected_percent, admission_threshold, frequency_decay_interval, short_frequency_decay_interval)) {
}

ggml_moe_cache_slots::~ggml_moe_cache_slots() = default;

ggml_moe_cache_slot_result ggml_moe_cache_slots::access(uint64_t key, size_t size_bytes, uint32_t group) {
    const size_t allocation_size = pimpl->aligned_size(size_bytes);
    auto policy_result = pimpl->policy.access(key, allocation_size, group);
    ++pimpl->stats.accesses;
    if (policy_result.hit) {
        ++pimpl->stats.hits;
    } else {
        ++pimpl->stats.misses;
        pimpl->stats.cost_rejections += policy_result.cost_rejected;
    }
    return pimpl->finalize(key, allocation_size, group, std::move(policy_result));
}

ggml_moe_cache_slot_result ggml_moe_cache_slots::prefetch(uint64_t key, size_t size_bytes, uint32_t group, uint64_t predicted_benefit) {
    const size_t allocation_size = pimpl->aligned_size(size_bytes);
    auto policy_result = pimpl->policy.prefetch(key, allocation_size, group, predicted_benefit);
    pimpl->stats.cost_rejections += policy_result.cost_rejected;
    return pimpl->finalize(key, allocation_size, group, std::move(policy_result));
}

void ggml_moe_cache_slots::set_group_count(uint32_t group_count) {
    pimpl->policy.set_group_count(group_count);
}

void ggml_moe_cache_slots::set_group_budgets(const std::vector<size_t> & budgets) {
    pimpl->policy.set_group_budgets(budgets);
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

struct ggml_moe_prefetch_control::impl {
    uint32_t useful = 0;
    uint32_t wasted = 0;
    uint32_t updates = 0;
};

ggml_moe_prefetch_control::ggml_moe_prefetch_control() : pimpl(new impl) {
}

ggml_moe_prefetch_control::~ggml_moe_prefetch_control() = default;

void ggml_moe_prefetch_control::record_outcome(bool useful) {
    if (useful) {
        ++pimpl->useful;
    } else {
        ++pimpl->wasted;
    }
    if (pimpl->useful + pimpl->wasted >= 64) {
        pimpl->useful /= 2;
        pimpl->wasted /= 2;
    }
}

ggml_moe_prefetch_limits ggml_moe_prefetch_control::limits(
        size_t base_bytes,
        bool copy_backlog,
        uint32_t evictions,
        uint32_t remaining_tokens) {
    ++pimpl->updates;
    if (base_bytes == 0 || remaining_tokens <= 1) {
        return {};
    }

    ggml_moe_prefetch_limits result { base_bytes, 4 };
    if (copy_backlog) {
        result.max_bytes /= 4;
    }
    const uint32_t outcomes = pimpl->useful + pimpl->wasted;
    if (outcomes < 8) {
        result.max_bytes /= 2;
    } else if ((uint64_t) pimpl->useful * 2 < outcomes) {
        if ((uint64_t) pimpl->useful * 4 >= outcomes) {
            result.max_bytes /= 2;
            result.min_probability_percent = 8;
        } else if ((uint64_t) pimpl->useful * 8 >= outcomes) {
            result.max_bytes /= 4;
            result.min_probability_percent = 12;
        } else if (pimpl->updates % 16 == 0) {
            result.max_bytes /= 8;
            result.min_probability_percent = 20;
        } else {
            return {};
        }
    }

    if (evictions >= 4) {
        result.max_bytes /= 4;
    } else if (evictions > 0) {
        result.max_bytes /= 2;
    }
    if (remaining_tokens <= 4) {
        result.max_bytes /= 4;
        result.min_probability_percent = std::max(result.min_probability_percent, 20u);
    } else if (remaining_tokens <= 16) {
        result.max_bytes /= 2;
        result.min_probability_percent = std::max(result.min_probability_percent, 8u);
    }
    return result;
}

uint32_t ggml_moe_prefetch_control::useful() const {
    return pimpl->useful;
}

uint32_t ggml_moe_prefetch_control::wasted() const {
    return pimpl->wasted;
}

uint32_t ggml_moe_prefetch_control::updates() const {
    return pimpl->updates;
}

struct ggml_moe_markov::impl {
    impl(uint32_t state_count, uint32_t prediction_count) :
        state_count(state_count),
        prediction_count(prediction_count),
        transitions((size_t) state_count * state_count, 0),
        totals(state_count, 0) {
    }

    uint32_t state_count;
    uint32_t prediction_count;
    std::vector<uint16_t> transitions;
    std::vector<uint32_t> totals;
    std::vector<uint32_t> previous;
};

ggml_moe_markov::ggml_moe_markov(uint32_t state_count, uint32_t prediction_count) :
    pimpl(new impl(state_count, prediction_count)) {
}

ggml_moe_markov::~ggml_moe_markov() = default;

void ggml_moe_markov::seed(const uint32_t * transitions, uint32_t prior_strength) {
    std::fill(pimpl->transitions.begin(), pimpl->transitions.end(), 0);
    std::fill(pimpl->totals.begin(), pimpl->totals.end(), 0);
    pimpl->previous.clear();
    if (!transitions || prior_strength == 0) {
        return;
    }

    for (uint32_t source = 0; source < pimpl->state_count; ++source) {
        uint64_t total = 0;
        for (uint32_t destination = 0; destination < pimpl->state_count; ++destination) {
            total += transitions[(size_t) source * pimpl->state_count + destination];
        }
        if (total == 0) {
            continue;
        }
        for (uint32_t destination = 0; destination < pimpl->state_count; ++destination) {
            const uint32_t count = transitions[(size_t) source * pimpl->state_count + destination];
            if (count == 0) {
                continue;
            }
            const uint32_t scaled = (uint32_t) (((uint64_t) count * prior_strength + total / 2) / total);
            if (scaled == 0) {
                continue;
            }
            pimpl->transitions[(size_t) source * pimpl->state_count + destination] =
                    (uint16_t) std::min<uint32_t>(scaled, std::numeric_limits<uint16_t>::max());
            pimpl->totals[source] += scaled;
        }
    }
}

std::vector<ggml_moe_markov_prediction> ggml_moe_markov::predict(const std::vector<uint32_t> & states) const {
    std::vector<ggml_moe_markov_prediction> ranked;
    ranked.reserve(pimpl->state_count);
    uint64_t total = 0;
    for (uint32_t current : states) {
        if (current < pimpl->state_count) {
            total += pimpl->totals[current];
        }
    }
    const uint32_t total_clamped = (uint32_t) std::min<uint64_t>(total, std::numeric_limits<uint32_t>::max());
    for (uint32_t predicted = 0; predicted < pimpl->state_count; ++predicted) {
        uint64_t count = 0;
        for (uint32_t current : states) {
            if (current < pimpl->state_count) {
                count += pimpl->transitions[(size_t) current * pimpl->state_count + predicted];
            }
        }
        if (count > 0) {
            ranked.push_back({
                predicted,
                (uint32_t) std::min<uint64_t>(count, std::numeric_limits<uint32_t>::max()),
                total_clamped,
            });
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const ggml_moe_markov_prediction & a, const ggml_moe_markov_prediction & b) {
        const uint64_t a_score = (uint64_t) a.count * b.total;
        const uint64_t b_score = (uint64_t) b.count * a.total;
        return a_score != b_score ? a_score > b_score : a.state < b.state;
    });
    if (ranked.size() > pimpl->prediction_count) {
        ranked.resize(pimpl->prediction_count);
    }
    return ranked;
}

std::vector<ggml_moe_markov_prediction> ggml_moe_markov::observe(const std::vector<uint32_t> & states) {
    if (states.empty()) {
        return {};
    }
    const uint32_t valid = (uint32_t) std::count_if(states.begin(), states.end(), [&](uint32_t state) {
        return state < pimpl->state_count;
    });
    for (uint32_t previous : pimpl->previous) {
        if (previous >= pimpl->state_count) {
            continue;
        }
        if (pimpl->totals[previous] > 32768) {
            for (uint32_t next = 0; next < pimpl->state_count; ++next) {
                pimpl->transitions[(size_t) previous * pimpl->state_count + next] /= 2;
            }
            pimpl->totals[previous] /= 2;
        }
        for (uint32_t current : states) {
            if (current >= pimpl->state_count) {
                continue;
            }
            uint16_t & count = pimpl->transitions[(size_t) previous * pimpl->state_count + current];
            if (count != std::numeric_limits<uint16_t>::max()) {
                ++count;
            }
        }
        if (pimpl->totals[previous] <= std::numeric_limits<uint32_t>::max() - valid) {
            pimpl->totals[previous] += valid;
        } else {
            pimpl->totals[previous] = std::numeric_limits<uint32_t>::max();
        }
    }

    auto ranked = predict(states);
    pimpl->previous = states;
    return ranked;
}
