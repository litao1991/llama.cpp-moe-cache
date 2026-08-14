#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml-moe-cache.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

static bool contains(const std::vector<uint64_t> & keys, uint64_t key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

static void test_frequency_admission() {
    ggml_moe_cache_policy cache(3);

    auto result = cache.access(1, 1);
    assert(!result.hit);
    assert(!result.admitted);
    assert(!cache.contains(1));

    result = cache.access(1, 1);
    assert(!result.hit);
    assert(result.admitted);
    assert(cache.contains(1));

    result = cache.access(1, 1);
    assert(result.hit);
    assert(!result.admitted);
    assert(cache.protected_bytes() == 1);
    assert(cache.probation_bytes() == 0);

    const auto & stats = cache.stats();
    assert(stats.accesses == 3);
    assert(stats.hits == 1);
    assert(stats.misses == 2);
    assert(stats.admission_attempts == 1);
    assert(stats.admissions == 1);
    assert(stats.evictions == 0);
}

static void test_slru_protects_reused_entries() {
    ggml_moe_cache_policy cache(4, 50, 1, 4096, 1);

    cache.access(1, 1);
    cache.access(2, 1);
    cache.access(3, 1);
    cache.access(4, 1);
    cache.access(1, 1);
    cache.access(2, 1);
    cache.access(3, 1);

    assert(cache.contains(1));
    assert(cache.contains(2));
    assert(cache.contains(3));
    assert(cache.contains(4));
    assert(cache.protected_bytes() == 2);
    assert(cache.probation_bytes() == 2);

    const auto result = cache.access(5, 1);
    assert(result.admitted);
    assert(result.evicted.size() == 1);
    assert(result.evicted[0] == 4);
    assert(cache.contains(1));
    assert(cache.contains(2));
    assert(cache.contains(3));
    assert(!cache.contains(4));
    assert(cache.contains(5));
}

static void test_weighted_eviction() {
    ggml_moe_cache_policy cache(10, 80, 1, 4096, 1);

    cache.access(1, 6);
    cache.access(2, 4);
    cache.access(1, 6);

    const auto result = cache.access(3, 5);
    assert(result.admitted);
    assert(result.evicted.size() == 2);
    assert(contains(result.evicted, 1));
    assert(contains(result.evicted, 2));
    assert(cache.resident_bytes() == 5);
    assert(cache.contains(3));
}

static void test_oversize_and_erase() {
    ggml_moe_cache_policy cache(8, 80, 1);

    auto result = cache.access(1, 9);
    assert(!result.admitted);
    assert(cache.resident_bytes() == 0);

    result = cache.access(2, 8);
    assert(result.admitted);
    assert(cache.erase(2));
    assert(!cache.erase(2));
    assert(cache.resident_bytes() == 0);
    assert(cache.size() == 0);
}

static void test_slot_alignment_and_coalescing() {
    ggml_moe_cache_slots cache(16, 4, 80, 1);

    auto result = cache.access(1, 5);
    assert(result.admitted);
    assert(result.offset == 0);
    assert(cache.resident_bytes() == 8);

    result = cache.access(2, 3);
    assert(result.admitted);
    assert(result.offset == 8);

    result = cache.access(3, 4);
    assert(result.admitted);
    assert(result.offset == 12);

    assert(cache.erase(2));
    assert(cache.erase(3));

    result = cache.access(4, 7);
    assert(result.admitted);
    assert(result.offset == 8);
    assert(cache.offset(4) == 8);
}

static void test_slot_fragmentation_recovery() {
    ggml_moe_cache_slots cache(16, 1, 80, 1);

    cache.access(1, 6);
    cache.access(2, 6);
    cache.access(3, 4);
    assert(cache.erase(1));
    assert(cache.erase(3));
    assert(cache.largest_free_range() == 6);

    const auto result = cache.access(4, 8);
    assert(result.admitted);
    assert(contains(result.evicted, 2));
    assert(cache.resident_entries() == 1);
    assert(cache.resident_bytes() == 8);
    assert(cache.largest_free_range() == 8);

    const auto & stats = cache.stats();
    assert(stats.admission_attempts == 4);
    assert(stats.admissions == 4);
    assert(stats.allocation_failures == 0);
    assert(stats.evictions == 1);
}

static void test_group_soft_budget() {
    ggml_moe_cache_slots cache(12, 1, 80, 1, 4096, 1);
    cache.set_group_count(3);

    cache.access(1, 4, 1);
    cache.access(2, 4, 0);
    cache.access(3, 4, 0);
    const auto result = cache.access(4, 4, 2);

    assert(result.admitted);
    assert(result.evicted.size() == 1);
    assert(result.evicted[0] == 2);
    assert(cache.contains(1));
    assert(!cache.contains(2));
    assert(cache.contains(3));
    assert(cache.contains(4));
    assert(cache.group_resident_bytes(0) == 4);
    assert(cache.group_resident_bytes(1) == 4);
    assert(cache.group_resident_bytes(2) == 4);
}

static void test_dynamic_group_budget() {
    ggml_moe_cache_slots cache(12, 1, 80, 1, 4096, 1);
    cache.set_group_budgets({ 2, 6, 4 });

    cache.prefetch(1, 2, 0);
    cache.prefetch(2, 2, 0);
    cache.prefetch(3, 6, 1);
    cache.prefetch(4, 2, 2);
    const auto result = cache.prefetch(5, 2, 2);

    assert(result.admitted);
    assert(result.evicted.size() == 1);
    assert(result.evicted[0] == 1);
    assert(cache.group_resident_bytes(0) == 2);
    assert(cache.group_resident_bytes(1) == 6);
    assert(cache.group_resident_bytes(2) == 4);
}

static void test_prefetch_protects_active_entries() {
    ggml_moe_cache_slots cache(12, 1, 80, 1, 4096, 1);

    cache.prefetch(1, 4);
    cache.prefetch(2, 4);
    cache.prefetch(3, 4);
    const auto result = cache.prefetch(4, 4, UINT32_MAX, UINT64_MAX, { 2, 3 });

    assert(result.admitted);
    assert(result.evicted.size() == 1);
    assert(result.evicted[0] == 1);
    assert(cache.contains(2));
    assert(cache.contains(3));
    assert(cache.contains(4));
}

static void test_layer_budget_waterfill() {
    const std::vector<ggml_moe_layer_candidate> candidates = {
        { 0, 2, 100, 2 },
        { 0, 2, 90, 2 },
        { 0, 2, 80, 2 },
        { 1, 2, 60, 2 },
        { 1, 2, 10, 2 },
        { 1, 2, 5, 2 },
    };
    const auto budgets = ggml_moe_layer_budgets(8, 2, candidates);
    assert(budgets.size() == 2);
    assert(budgets[0] == 6);
    assert(budgets[1] == 2);
}

static void test_mobile_capacity_budget() {
    const size_t mib = 1024 * 1024;
    assert(ggml_moe_cache_capacity_budget(
            1024 * mib, 900 * mib, 700 * mib, 512 * mib, 128 * mib, 32 * mib, 256 * mib) == 256 * mib);
    assert(ggml_moe_cache_capacity_budget(
            1024 * mib, 900 * mib, 300 * mib, 512 * mib, 128 * mib, 32 * mib, 256 * mib) == 160 * mib);
    assert(ggml_moe_cache_capacity_budget(
            1024 * mib, 120 * mib, 700 * mib, 512 * mib, 128 * mib, 32 * mib, 256 * mib) == 0);
}

static void test_cost_aware_admission() {
    ggml_moe_cache_policy cache(8, 80, 2, 0, 0);

    cache.access(1, 8);
    cache.access(1, 8);
    for (int i = 0; i < 6; ++i) {
        cache.access(1, 8);
    }

    cache.access(2, 8);
    const auto rejected = cache.access(2, 8);
    assert(rejected.cost_rejected);
    assert(!rejected.admitted);
    assert(cache.contains(1));
    assert(!cache.contains(2));
    assert(cache.stats().cost_rejections == 1);

    for (int i = 0; i < 12; ++i) {
        cache.access(2, 8);
    }
    assert(cache.contains(2));
    assert(!cache.contains(1));
}

static void test_full_eviction_set_cost() {
    ggml_moe_cache_policy cache(10, 80, 1, 0, 0);

    cache.prefetch(1, 4);
    cache.prefetch(2, 6);
    for (int i = 0; i < 8; ++i) {
        cache.access(2, 6);
    }

    const auto rejected = cache.prefetch(3, 7, UINT32_MAX, 1024);
    assert(rejected.cost_rejected);
    assert(!rejected.admitted);
    assert(cache.contains(1));
    assert(cache.contains(2));
    assert(cache.stats().cost_rejections == 1);
}

static void test_frequency_windows() {
    ggml_moe_cache_policy cache(8, 80, 2, 8, 2);

    cache.access(1, 1);
    cache.access(1, 1);
    assert(cache.short_frequency(1) == 1);
    assert(cache.frequency(1) == 2);
    cache.access(1, 1);
    cache.access(1, 1);
    assert(cache.short_frequency(1) == 2);
    assert(cache.frequency(1) == 4);
}

static void test_markov_prediction() {
    ggml_moe_markov markov(4, 2);

    assert(markov.observe({ 1 }).empty());
    assert(markov.observe({ 2 }).empty());
    auto predictions = markov.observe({ 1 });
    assert(predictions.size() == 1);
    assert(predictions[0].state == 2);
    assert(predictions[0].count == 1);
    assert(predictions[0].total == 1);
    markov.observe({ 3 });
    predictions = markov.observe({ 1 });
    assert(predictions.size() == 2);
    assert(predictions[0].state == 2);
    assert(predictions[1].state == 3);
    assert(predictions[0].total == 2);
}

static void test_markov_prefill_seed() {
    ggml_moe_markov markov(4, 2);
    std::vector<uint32_t> transitions(16, 0);
    transitions[1 * 4 + 2] = 6;
    transitions[1 * 4 + 3] = 2;
    markov.seed(transitions.data(), 16);

    auto predictions = markov.predict({ 1 });
    assert(predictions.size() == 2);
    assert(predictions[0].state == 2);
    assert(predictions[0].count == 12);
    assert(predictions[0].total == 16);
    assert(predictions[1].state == 3);

    predictions = markov.observe({ 1 });
    assert(predictions.size() == 2);
    assert(predictions[0].state == 2);
}

static void test_markov_sparse_prefill_seed() {
    ggml_moe_markov markov(4, 2);
    std::vector<uint32_t> transitions(4 * 2, 0);
    transitions[1 * 2 + 0] = (6u << 16) | (2u + 1);
    transitions[1 * 2 + 1] = (2u << 16) | (3u + 1);
    markov.seed_sparse(transitions.data(), 2, 16);

    const auto predictions = markov.predict({ 1 });
    assert(predictions.size() == 2);
    assert(predictions[0].state == 2);
    assert(predictions[0].count == 12);
    assert(predictions[0].total == 16);
    assert(predictions[1].state == 3);
}

static void test_markov_cross_layer_transition() {
    ggml_moe_markov markov(8, 2);

    markov.observe_transition({ 1, 2 }, { 4, 5 });
    markov.observe_transition({ 1 }, { 4 });
    const auto predictions = markov.predict({ 1 });

    assert(predictions.size() == 2);
    assert(predictions[0].state == 4);
    assert(predictions[0].count == 2);
    assert(predictions[0].total == 3);
    assert(predictions[1].state == 5);
}

static void test_prefetch_does_not_count_as_access() {
    ggml_moe_cache_slots cache(4, 1, 80, 2);

    auto result = cache.prefetch(1, 4);
    assert(result.admitted);
    assert(cache.contains(1));
    assert(cache.stats().accesses == 0);
    assert(cache.stats().admissions == 1);

    result = cache.prefetch(1, 4, UINT32_MAX, 0);
    assert(result.hit);
    assert(!result.cost_rejected);
    assert(cache.stats().accesses == 0);
    assert(cache.stats().admissions == 1);

    result = cache.access(1, 4);
    assert(result.hit);
    assert(cache.stats().accesses == 1);
    assert(cache.stats().hits == 1);
}

static void test_required_set_is_not_evicted() {
    ggml_moe_cache_slots cache(12, 1, 80, 1, 0, 0);
    assert(cache.prefetch(1, 4).admitted);
    assert(cache.prefetch(2, 4).admitted);
    assert(cache.prefetch(3, 4).admitted);

    const std::vector<uint64_t> required = { 1, 4, 5 };
    auto result = cache.require(1, 4, 0, required);
    assert(result.hit);
    result = cache.require(4, 4, 0, required);
    assert(result.admitted);
    assert(cache.contains(1));
    result = cache.require(5, 4, 0, required);
    assert(result.admitted);
    assert(cache.contains(1));
    assert(cache.contains(4));
    assert(cache.contains(5));
}

static void test_adaptive_prefetch_control() {
    ggml_moe_prefetch_control control;

    auto limits = control.limits(1024, false, 0);
    assert(limits.max_bytes == 512);
    assert(limits.min_probability_percent == 4);

    for (int i = 0; i < 8; ++i) {
        control.record_outcome(true);
    }
    limits = control.limits(1024, false, 0);
    assert(limits.max_bytes == 1024);
    assert(limits.min_probability_percent == 4);

    limits = control.limits(1024, true, 0);
    assert(limits.max_bytes == 256);
    assert(limits.min_probability_percent == 4);
    limits = control.limits(1024, false, 4);
    assert(limits.max_bytes == 256);
    assert(limits.min_probability_percent == 4);
    limits = control.limits(1024, false, 0, 4);
    assert(limits.max_bytes == 256);
    assert(limits.min_probability_percent == 20);
}

int main() {
    test_frequency_admission();
    test_slru_protects_reused_entries();
    test_weighted_eviction();
    test_oversize_and_erase();
    test_slot_alignment_and_coalescing();
    test_slot_fragmentation_recovery();
    test_group_soft_budget();
    test_dynamic_group_budget();
    test_prefetch_protects_active_entries();
    test_layer_budget_waterfill();
    test_mobile_capacity_budget();
    test_cost_aware_admission();
    test_full_eviction_set_cost();
    test_frequency_windows();
    test_markov_prediction();
    test_markov_prefill_seed();
    test_markov_sparse_prefill_seed();
    test_markov_cross_layer_transition();
    test_prefetch_does_not_count_as_access();
    test_required_set_is_not_evicted();
    test_adaptive_prefetch_control();
    return 0;
}
