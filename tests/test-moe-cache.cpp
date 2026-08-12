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
    ggml_moe_cache_policy cache(4, 50, 1);

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
    ggml_moe_cache_policy cache(10, 80, 1);

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
    ggml_moe_cache_slots cache(12, 1, 80, 1);
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

int main() {
    test_frequency_admission();
    test_slru_protects_reused_entries();
    test_weighted_eviction();
    test_oversize_and_erase();
    test_slot_alignment_and_coalescing();
    test_slot_fragmentation_recovery();
    test_group_soft_budget();
    return 0;
}
