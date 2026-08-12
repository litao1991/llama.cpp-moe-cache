#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_moe_cache_stats {
    uint64_t accesses            = 0;
    uint64_t hits                = 0;
    uint64_t misses              = 0;
    uint64_t admission_attempts  = 0;
    uint64_t admissions          = 0;
    uint64_t allocation_failures = 0;
    uint64_t evictions           = 0;
};

struct ggml_moe_cache_result {
    bool hit      = false;
    bool admitted = false;
    std::vector<uint64_t> evicted;
};

struct ggml_moe_cache_slot_result {
    bool hit      = false;
    bool admitted = false;
    size_t offset = SIZE_MAX;
    std::vector<uint64_t> evicted;
};

class ggml_moe_cache_policy {
public:
    ggml_moe_cache_policy(
            size_t capacity_bytes,
            uint32_t protected_percent = 80,
            uint32_t admission_threshold = 2,
            uint64_t frequency_decay_interval = 4096);
    ~ggml_moe_cache_policy();

    ggml_moe_cache_policy(const ggml_moe_cache_policy &) = delete;
    ggml_moe_cache_policy & operator=(const ggml_moe_cache_policy &) = delete;

    ggml_moe_cache_result access(uint64_t key, size_t size_bytes, uint32_t group = UINT32_MAX);
    bool erase(uint64_t key);
    bool evict(uint64_t & key);
    void set_group_count(uint32_t group_count);

    bool contains(uint64_t key) const;
    size_t capacity_bytes() const;
    size_t resident_bytes() const;
    size_t protected_bytes() const;
    size_t probation_bytes() const;
    size_t group_resident_bytes(uint32_t group) const;
    size_t size() const;
    uint32_t frequency(uint64_t key) const;
    const ggml_moe_cache_stats & stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class ggml_moe_cache_slots {
public:
    ggml_moe_cache_slots(
            size_t capacity_bytes,
            size_t alignment,
            uint32_t protected_percent = 80,
            uint32_t admission_threshold = 2,
            uint64_t frequency_decay_interval = 4096);
    ~ggml_moe_cache_slots();

    ggml_moe_cache_slots(const ggml_moe_cache_slots &) = delete;
    ggml_moe_cache_slots & operator=(const ggml_moe_cache_slots &) = delete;

    ggml_moe_cache_slot_result access(uint64_t key, size_t size_bytes, uint32_t group = UINT32_MAX);
    bool erase(uint64_t key);
    void set_group_count(uint32_t group_count);
    void reset_stats();

    bool contains(uint64_t key) const;
    size_t offset(uint64_t key) const;
    size_t resident_bytes() const;
    size_t resident_entries() const;
    size_t largest_free_range() const;
    size_t group_resident_bytes(uint32_t group) const;
    const ggml_moe_cache_stats & stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
