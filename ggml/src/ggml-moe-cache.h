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
    uint64_t cost_rejections     = 0;
    uint64_t allocation_failures = 0;
    uint64_t evictions           = 0;
};

struct ggml_moe_cache_result {
    bool hit           = false;
    bool admitted      = false;
    bool cost_rejected = false;
    std::vector<uint64_t> evicted;
};

struct ggml_moe_cache_slot_result {
    bool hit           = false;
    bool admitted      = false;
    bool cost_rejected = false;
    size_t offset = SIZE_MAX;
    std::vector<uint64_t> evicted;
};

struct ggml_moe_layer_candidate {
    uint32_t layer;
    size_t size_bytes;
    uint64_t expected_hits;
    size_t read_bytes;
};

std::vector<size_t> ggml_moe_layer_budgets(
        size_t capacity_bytes,
        uint32_t layer_count,
        const std::vector<ggml_moe_layer_candidate> & candidates);

size_t ggml_moe_cache_capacity_budget(
        size_t requested,
        size_t metal_headroom,
        size_t process_headroom,
        size_t device_cap,
        size_t reserve,
        size_t alignment,
        size_t initial_cap);

class ggml_moe_cache_policy {
public:
    ggml_moe_cache_policy(
            size_t capacity_bytes,
            uint32_t protected_percent = 80,
            uint32_t admission_threshold = 2,
            uint64_t frequency_decay_interval = 4096,
            uint64_t short_frequency_decay_interval = 256);
    ~ggml_moe_cache_policy();

    ggml_moe_cache_policy(const ggml_moe_cache_policy &) = delete;
    ggml_moe_cache_policy & operator=(const ggml_moe_cache_policy &) = delete;

    ggml_moe_cache_result access(uint64_t key, size_t size_bytes, uint32_t group = UINT32_MAX);
    ggml_moe_cache_result prefetch(uint64_t key, size_t size_bytes, uint32_t group = UINT32_MAX, uint64_t predicted_benefit = UINT64_MAX);
    ggml_moe_cache_result require(uint64_t key, size_t size_bytes, uint32_t group, const std::vector<uint64_t> & protected_keys);
    bool erase(uint64_t key);
    bool evict(uint64_t & key);
    bool evict(uint64_t & key, const std::vector<uint64_t> & protected_keys);
    void set_group_count(uint32_t group_count);
    void set_group_budgets(const std::vector<size_t> & budgets);

    bool contains(uint64_t key) const;
    size_t capacity_bytes() const;
    size_t resident_bytes() const;
    size_t protected_bytes() const;
    size_t probation_bytes() const;
    size_t group_resident_bytes(uint32_t group) const;
    size_t size() const;
    uint32_t frequency(uint64_t key) const;
    uint32_t short_frequency(uint64_t key) const;
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
            uint64_t frequency_decay_interval = 4096,
            uint64_t short_frequency_decay_interval = 256);
    ~ggml_moe_cache_slots();

    ggml_moe_cache_slots(const ggml_moe_cache_slots &) = delete;
    ggml_moe_cache_slots & operator=(const ggml_moe_cache_slots &) = delete;

    ggml_moe_cache_slot_result access(uint64_t key, size_t size_bytes, uint32_t group = UINT32_MAX);
    ggml_moe_cache_slot_result prefetch(uint64_t key, size_t size_bytes, uint32_t group = UINT32_MAX, uint64_t predicted_benefit = UINT64_MAX);
    ggml_moe_cache_slot_result require(uint64_t key, size_t size_bytes, uint32_t group, const std::vector<uint64_t> & protected_keys);
    bool erase(uint64_t key);
    void set_group_count(uint32_t group_count);
    void set_group_budgets(const std::vector<size_t> & budgets);
    void reset_stats();
    void reset(size_t capacity_bytes, size_t alignment);

    bool contains(uint64_t key) const;
    size_t offset(uint64_t key) const;
    size_t resident_bytes() const;
    size_t resident_entries() const;
    size_t largest_free_range() const;
    size_t group_resident_bytes(uint32_t group) const;
    uint32_t frequency(uint64_t key) const;
    uint32_t short_frequency(uint64_t key) const;
    const ggml_moe_cache_stats & stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct ggml_moe_markov_prediction {
    uint32_t state;
    uint32_t count;
    uint32_t total;
};

struct ggml_moe_prefetch_limits {
    size_t max_bytes = 0;
    uint32_t min_probability_percent = 100;
};

class ggml_moe_prefetch_control {
public:
    ggml_moe_prefetch_control();
    ~ggml_moe_prefetch_control();

    ggml_moe_prefetch_control(const ggml_moe_prefetch_control &) = delete;
    ggml_moe_prefetch_control & operator=(const ggml_moe_prefetch_control &) = delete;

    void record_outcome(bool useful);
    ggml_moe_prefetch_limits limits(size_t base_bytes, bool copy_backlog, uint32_t evictions, uint32_t remaining_tokens = UINT32_MAX);
    uint32_t useful() const;
    uint32_t wasted() const;
    uint32_t updates() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class ggml_moe_markov {
public:
    ggml_moe_markov(uint32_t state_count, uint32_t prediction_count = 2);
    ~ggml_moe_markov();

    ggml_moe_markov(const ggml_moe_markov &) = delete;
    ggml_moe_markov & operator=(const ggml_moe_markov &) = delete;

    void seed(const uint32_t * transitions, uint32_t prior_strength = 32);
    void seed_sparse(const uint32_t * transitions, uint32_t transition_slots, uint32_t prior_strength = 32);
    std::vector<ggml_moe_markov_prediction> predict(const std::vector<uint32_t> & states) const;
    std::vector<ggml_moe_markov_prediction> observe(const std::vector<uint32_t> & states);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
