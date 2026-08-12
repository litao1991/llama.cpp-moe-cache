#include "ggml-metal-moe-cache.h"
#include "ggml-metal-moe-profile.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-moe-cache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#import <Metal/Metal.h>

static constexpr size_t GGML_METAL_MOE_STAGING_SIZE = 32 * 1024 * 1024;

static bool ggml_metal_moe_env_enabled(const char * name, bool default_value) {
    const char * value = getenv(name);
    return value ? strcmp(value, "0") != 0 : default_value;
}

static uint32_t ggml_metal_moe_env_u32(const char * name, uint32_t default_value) {
    const char * value = getenv(name);
    if (!value) {
        return default_value;
    }
    char * end = nullptr;
    const unsigned long parsed = strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed <= UINT32_MAX ? (uint32_t) parsed : default_value;
}

struct ggml_metal_moe_tensor_cache {
    ~ggml_metal_moe_tensor_cache() {
        if (page_table_buffer) {
            ggml_metal_buffer_free(page_table_buffer);
        }
        if (access_buffer) {
            ggml_metal_buffer_free(access_buffer);
        }
    }

    const ggml_tensor * tensor = nullptr;
    uint32_t id = 0;
    uint32_t layer = UINT32_MAX;
    uint32_t n_experts = 0;
    size_t expert_size = 0;
    ggml_metal_buffer_t page_table_buffer = nullptr;
    ggml_metal_buffer_t access_buffer = nullptr;
    uint64_t * page_table = nullptr;
    uint32_t * access = nullptr;
    std::unique_ptr<ggml_moe_markov> markov;
};

struct ggml_metal_moe_cache {
    ggml_metal_moe_cache(ggml_metal_device_t dev, size_t capacity_bytes) :
        dev(dev),
        capacity(capacity_bytes),
        slots(capacity_bytes, 4096) {
        prefetch_enabled = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_PREFETCH", true);
        adaptive_prefetch = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_ADAPTIVE_PREFETCH", true);
        prefill_markov = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_PREFILL_MARKOV", true);
        remaining_tokens_hint = ggml_metal_moe_env_u32("GGML_METAL_MOE_CACHE_REMAINING_TOKENS", UINT32_MAX);
        buffer = ggml_metal_buffer_init(dev, capacity_bytes, false);
        for (size_t i = 0; i < 2; ++i) {
            staging[i] = ggml_metal_buffer_init(dev, GGML_METAL_MOE_STAGING_SIZE, true);
        }
    }

    ~ggml_metal_moe_cache() {
        wait_for_copy();
        wasted_prefetches += prefetched.size();
        for (size_t i = 0; i < prefetched.size(); ++i) {
            prefetch_control.record_outcome(false);
        }
        prefetched.clear();
        const auto & stats = slots.stats();
        GGML_LOG_INFO("%s: accesses = %llu, hits = %llu, admission attempts = %llu, admissions = %llu, cost rejections = %llu, allocation failures = %llu, evictions = %llu, resident entries = %zu, resident bytes = %zu, staging batches = %llu, async copies = %llu, predictions = %llu, prefetch attempts = %llu, prefetch admissions = %llu, useful prefetches = %llu, wasted prefetches = %llu, adaptive skips = %llu, copy pressure updates = %llu, remaining stops = %llu, controller outcomes = %u/%u, controller updates = %u, prefill Markov tensors = %zu\n",
                __func__,
                (unsigned long long) stats.accesses,
                (unsigned long long) stats.hits,
                (unsigned long long) stats.admission_attempts,
                (unsigned long long) stats.admissions,
                (unsigned long long) stats.cost_rejections,
                (unsigned long long) stats.allocation_failures,
                (unsigned long long) stats.evictions,
                slots.resident_entries(),
                slots.resident_bytes(),
                (unsigned long long) staging_batches,
                (unsigned long long) async_copies,
                (unsigned long long) predictions,
                (unsigned long long) prefetch_attempts,
                (unsigned long long) prefetch_admissions,
                (unsigned long long) useful_prefetches,
                (unsigned long long) wasted_prefetches,
                (unsigned long long) adaptive_skips,
                (unsigned long long) copy_pressure_updates,
                (unsigned long long) remaining_stops,
                prefetch_control.useful(),
                prefetch_control.wasted(),
                prefetch_control.updates(),
                prefill_markov_tensors);
        tensors.clear();
        for (auto & item : staging) {
            if (item) {
                ggml_metal_buffer_free(item);
            }
        }
        if (buffer) {
            ggml_metal_buffer_free(buffer);
        }
    }

    ggml_metal_device_t dev;
    size_t capacity;
    ggml_metal_buffer_t buffer = nullptr;
    ggml_moe_cache_slots slots;
    std::mutex mutex;
    std::unordered_map<const ggml_tensor *, ggml_metal_moe_tensor_cache *> tensor_map;
    std::vector<std::unique_ptr<ggml_metal_moe_tensor_cache>> tensors;
    uint32_t n_layers = 0;
    size_t preloaded_entries = 0;
    size_t preloaded_bytes = 0;
    ggml_metal_buffer_t staging[2] = { nullptr, nullptr };
    size_t staging_index = 0;
    id<MTLCommandBuffer> staging_cmd_buf[2] = { nil, nil };
    id<MTLCommandBuffer> copy_cmd_buf = nil;
    id<MTLBlitCommandEncoder> copy_encoder = nil;
    size_t staged_bytes = 0;
    uint64_t staging_batches = 0;
    uint64_t async_copies = 0;
    uint64_t predictions = 0;
    uint64_t prefetch_attempts = 0;
    uint64_t prefetch_admissions = 0;
    uint64_t useful_prefetches = 0;
    uint64_t wasted_prefetches = 0;
    uint64_t adaptive_skips = 0;
    uint64_t copy_pressure_updates = 0;
    uint64_t remaining_stops = 0;
    uint64_t decode_updates = 0;
    size_t prefill_markov_tensors = 0;
    bool prefetch_enabled = true;
    bool adaptive_prefetch = true;
    bool prefill_markov = true;
    uint32_t remaining_tokens_hint = UINT32_MAX;
    ggml_moe_prefetch_control prefetch_control;
    std::unordered_set<uint64_t> prefetched;

    bool copy_backlog() const {
        uint32_t in_flight = 0;
        for (id<MTLCommandBuffer> item : staging_cmd_buf) {
            if (item && [item status] < MTLCommandBufferStatusCompleted) {
                ++in_flight;
            }
        }
        return in_flight == 2 || (in_flight > 0 && staged_bytes >= GGML_METAL_MOE_STAGING_SIZE * 3 / 4);
    }

    void begin_copy() {
        if (copy_cmd_buf) {
            return;
        }
        if (staging_cmd_buf[staging_index]) {
            [staging_cmd_buf[staging_index] waitUntilCompleted];
            [staging_cmd_buf[staging_index] release];
            staging_cmd_buf[staging_index] = nil;
        }
        copy_cmd_buf = [(id<MTLCommandQueue>) ggml_metal_device_get_queue(dev) commandBuffer];
        [copy_cmd_buf retain];
        copy_encoder = [copy_cmd_buf blitCommandEncoder];
        staged_bytes = 0;
    }

    void commit_copy() {
        if (!copy_encoder) {
            return;
        }
        [copy_encoder endEncoding];
        [copy_cmd_buf commit];
        staging_cmd_buf[staging_index] = copy_cmd_buf;
        ++staging_batches;
        copy_cmd_buf = nil;
        copy_encoder = nil;
        staging_index = (staging_index + 1) % 2;
        staged_bytes = 0;
    }

    void wait_for_copy() {
        if (copy_encoder) {
            commit_copy();
        }
        for (auto & item : staging_cmd_buf) {
            if (item) {
                [item waitUntilCompleted];
                [item release];
                item = nil;
            }
        }
    }

    void copy_expert(const ggml_metal_moe_tensor_cache & entry, uint32_t expert, size_t offset) {
        size_t copied = 0;
        while (copied < entry.expert_size) {
            if (staged_bytes == GGML_METAL_MOE_STAGING_SIZE) {
                commit_copy();
            }
            begin_copy();
            const size_t chunk = std::min(entry.expert_size - copied, GGML_METAL_MOE_STAGING_SIZE - staged_bytes);
            memcpy((char *) ggml_metal_buffer_get_base(staging[staging_index]) + staged_bytes,
                    (const char *) entry.tensor->data + expert * entry.expert_size + copied, chunk);
            const auto src = ggml_metal_buffer_get_base_id(staging[staging_index]);
            const auto dst = ggml_metal_buffer_get_base_id(buffer);
            [copy_encoder copyFromBuffer:(id<MTLBuffer>) src.metal
                            sourceOffset:src.offs + staged_bytes
                                toBuffer:(id<MTLBuffer>) dst.metal
                       destinationOffset:dst.offs + offset + copied
                                    size:chunk];
            staged_bytes += chunk;
            copied += chunk;
            ++async_copies;
        }
    }

    void invalidate(uint64_t key) {
        if (prefetched.erase(key) > 0) {
            ++wasted_prefetches;
            prefetch_control.record_outcome(false);
        }
        const uint32_t tensor_id = (uint32_t) (key >> 32);
        const uint32_t expert_id = (uint32_t) key;
        if (tensor_id < tensors.size() && expert_id < tensors[tensor_id]->n_experts) {
            tensors[tensor_id]->page_table[expert_id] = UINT64_MAX;
        }
    }
};

struct ggml_metal_moe_preload_candidate {
    ggml_metal_moe_tensor_cache * entry;
    uint32_t expert;
    uint32_t count;
};

static uint64_t ggml_metal_moe_cache_key(uint32_t tensor_id, uint32_t expert) {
    return ((uint64_t) tensor_id << 32) | expert;
}

static uint32_t ggml_metal_moe_cache_layer(const char * name) {
    if (strncmp(name, "blk.", 4) != 0) {
        return UINT32_MAX;
    }
    char * end = nullptr;
    const unsigned long layer = strtoul(name + 4, &end, 10);
    if (end == name + 4 || *end != '.' || layer > UINT32_MAX) {
        return UINT32_MAX;
    }
    return (uint32_t) layer;
}

static struct ggml_metal_moe_cache_bindings ggml_metal_moe_cache_disabled(void) {
    return {
        false,
        { nullptr, 0 },
        { nullptr, 0 },
        { nullptr, 0 },
    };
}

ggml_metal_moe_cache_t ggml_metal_moe_cache_init(ggml_metal_device_t dev, size_t capacity_bytes) {
    std::unique_ptr<ggml_metal_moe_cache> cache(new ggml_metal_moe_cache(dev, capacity_bytes));
    if (!cache->buffer ||
            !cache->staging[0] || !cache->staging[1] ||
            !ggml_metal_buffer_is_shared(cache->staging[0]) || !ggml_metal_buffer_is_shared(cache->staging[1])) {
        return nullptr;
    }
    return cache.release();
}

void ggml_metal_moe_cache_free(ggml_metal_moe_cache_t cache) {
    delete cache;
}

struct ggml_metal_moe_cache_result ggml_metal_moe_cache_access(ggml_metal_moe_cache_t cache, uint64_t key, size_t size_bytes) {
    const auto result = cache->slots.access(key, size_bytes);
    return {
        result.hit,
        result.admitted,
        result.offset,
    };
}

struct ggml_metal_moe_cache_bindings ggml_metal_moe_cache_get_bindings(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * tensor) {
    if (!cache || !tensor || tensor->ne[2] <= 1 || tensor->ne[2] > std::numeric_limits<uint32_t>::max() || tensor->nb[2] > cache->capacity) {
        return ggml_metal_moe_cache_disabled();
    }

    ggml_backend_buffer_t backend_buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!backend_buffer) {
        return ggml_metal_moe_cache_disabled();
    }

    ggml_metal_buffer_t source_buffer = (ggml_metal_buffer_t) backend_buffer->context;
    if (!ggml_metal_buffer_is_shared(source_buffer)) {
        return ggml_metal_moe_cache_disabled();
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    const auto existing = cache->tensor_map.find(tensor);
    if (existing != cache->tensor_map.end()) {
        const auto * entry = existing->second;
        return {
            true,
            ggml_metal_buffer_get_base_id(cache->buffer),
            ggml_metal_buffer_get_base_id(entry->page_table_buffer),
            ggml_metal_buffer_get_base_id(entry->access_buffer),
        };
    }

    if (cache->tensors.size() >= std::numeric_limits<uint32_t>::max()) {
        return ggml_metal_moe_cache_disabled();
    }

    std::unique_ptr<ggml_metal_moe_tensor_cache> entry(new ggml_metal_moe_tensor_cache);
    entry->tensor = tensor;
    entry->id = (uint32_t) cache->tensors.size();
    entry->layer = ggml_metal_moe_cache_layer(tensor->name);
    if (entry->layer != UINT32_MAX) {
        cache->n_layers = std::max(cache->n_layers, entry->layer + 1);
        cache->slots.set_group_count(cache->n_layers);
    }
    entry->n_experts = (uint32_t) tensor->ne[2];
    entry->expert_size = tensor->nb[2];
    entry->page_table_buffer = ggml_metal_buffer_init(cache->dev, entry->n_experts * sizeof(uint64_t), true);
    entry->access_buffer = ggml_metal_buffer_init(cache->dev, entry->n_experts * sizeof(uint32_t), true);

    if (!entry->page_table_buffer || !entry->access_buffer ||
            !ggml_metal_buffer_is_shared(entry->page_table_buffer) ||
            !ggml_metal_buffer_is_shared(entry->access_buffer)) {
        return ggml_metal_moe_cache_disabled();
    }

    entry->page_table = (uint64_t *) ggml_metal_buffer_get_base(entry->page_table_buffer);
    entry->access = (uint32_t *) ggml_metal_buffer_get_base(entry->access_buffer);
    entry->markov.reset(new ggml_moe_markov(entry->n_experts, 8));
    std::fill(entry->page_table, entry->page_table + entry->n_experts, UINT64_MAX);
    std::fill(entry->access, entry->access + entry->n_experts, 0);

    ggml_metal_moe_tensor_cache * raw = entry.get();
    cache->tensor_map.emplace(tensor, raw);
    cache->tensors.emplace_back(std::move(entry));

    return {
        true,
        ggml_metal_buffer_get_base_id(cache->buffer),
        ggml_metal_buffer_get_base_id(raw->page_table_buffer),
        ggml_metal_buffer_get_base_id(raw->access_buffer),
    };
}

void ggml_metal_moe_cache_update(ggml_metal_moe_cache_t cache) {
    if (!cache) {
        return;
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    struct prediction {
        ggml_metal_moe_tensor_cache * entry;
        uint32_t expert;
        uint32_t count;
        uint32_t total;
    };
    std::vector<prediction> predictions;
    uint32_t demand_evictions = 0;
    bool observed_routes = false;

    for (const auto & entry_ptr : cache->tensors) {
        ggml_metal_moe_tensor_cache & entry = *entry_ptr;
        std::vector<uint32_t> current_experts;
        for (uint32_t expert = 0; expert < entry.n_experts; ++expert) {
            if (entry.access[expert] == 0) {
                continue;
            }
            entry.access[expert] = 0;
            current_experts.push_back(expert);

            const uint64_t key = ggml_metal_moe_cache_key(entry.id, expert);
            const auto result = cache->slots.access(key, entry.expert_size, entry.layer);
            if (result.hit && cache->prefetched.erase(key) > 0) {
                ++cache->useful_prefetches;
                cache->prefetch_control.record_outcome(true);
            }
            demand_evictions += (uint32_t) result.evicted.size();
            for (uint64_t evicted : result.evicted) {
                cache->invalidate(evicted);
            }

            if (result.admitted) {
                cache->copy_expert(entry, expert, result.offset);
                entry.page_table[expert] = result.offset;
            }
        }
        for (const auto & predicted : entry.markov->observe(current_experts)) {
            ++cache->predictions;
            const uint64_t key = ggml_metal_moe_cache_key(entry.id, predicted.state);
            if (predicted.count > 0 && predicted.total > 0 && !cache->slots.contains(key)) {
                predictions.push_back({ &entry, predicted.state, predicted.count, predicted.total });
            }
        }
        observed_routes = observed_routes || !current_experts.empty();
    }

    if (!observed_routes) {
        cache->commit_copy();
        return;
    }

    std::sort(predictions.begin(), predictions.end(), [](const prediction & a, const prediction & b) {
        const long double a_score = (long double) a.count / a.total / a.entry->expert_size;
        const long double b_score = (long double) b.count / b.total / b.entry->expert_size;
        return a_score != b_score ? a_score > b_score : a.count > b.count;
    });
    const size_t base_prefetch_limit = std::min<size_t>(8 * 1024 * 1024, cache->capacity / 64);
    ++cache->decode_updates;
    const uint32_t remaining_tokens = cache->remaining_tokens_hint == UINT32_MAX ? UINT32_MAX :
            (cache->remaining_tokens_hint > cache->decode_updates ? cache->remaining_tokens_hint - cache->decode_updates : 0);
    ggml_moe_prefetch_limits limits = { base_prefetch_limit, 4 };
    if (!cache->prefetch_enabled) {
        limits = {};
    } else if (cache->adaptive_prefetch) {
        const bool copy_backlog = cache->copy_backlog();
        cache->copy_pressure_updates += copy_backlog;
        cache->remaining_stops += remaining_tokens <= 1;
        limits = cache->prefetch_control.limits(base_prefetch_limit, copy_backlog, demand_evictions, remaining_tokens);
    }
    if (limits.max_bytes == 0 && !predictions.empty()) {
        ++cache->adaptive_skips;
    }
    size_t prefetch_bytes = 0;
    for (const auto & prediction : predictions) {
        const size_t allocation_size = (prediction.entry->expert_size + 4095) / 4096 * 4096;
        if ((uint64_t) prediction.count * 100 < (uint64_t) prediction.total * limits.min_probability_percent ||
                prefetch_bytes >= limits.max_bytes || allocation_size > limits.max_bytes - prefetch_bytes) {
            continue;
        }
        const uint64_t key = ggml_metal_moe_cache_key(prediction.entry->id, prediction.expert);
        if (cache->slots.contains(key)) {
            continue;
        }
        ++cache->prefetch_attempts;
        const auto result = cache->slots.prefetch(
                key,
                prediction.entry->expert_size,
                prediction.entry->layer,
                (uint64_t) prediction.count * 1024 / prediction.total);
        for (uint64_t evicted : result.evicted) {
            cache->invalidate(evicted);
        }
        if (result.admitted) {
            cache->copy_expert(*prediction.entry, prediction.expert, result.offset);
            prediction.entry->page_table[prediction.expert] = result.offset;
            cache->prefetched.insert(key);
            ++cache->prefetch_admissions;
            prefetch_bytes += allocation_size;
        }
    }
    cache->commit_copy();
}

void ggml_metal_moe_cache_preload(ggml_metal_moe_cache_t cache, ggml_metal_moe_profile_t profile) {
    if (!cache || !profile) {
        return;
    }

    const auto profile_stats = ggml_metal_moe_profile_get_stats(profile);
    std::vector<ggml_metal_moe_tensor_profile_stats> tensor_stats;
    tensor_stats.reserve(profile_stats.tensors);
    for (size_t i = 0; i < profile_stats.tensors; ++i) {
        ggml_metal_moe_tensor_profile_stats stats;
        if (ggml_metal_moe_profile_get_tensor_stats(profile, i, &stats)) {
            tensor_stats.push_back(stats);
            ggml_metal_moe_cache_get_bindings(cache, stats.tensor);
        }
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    std::vector<ggml_metal_moe_preload_candidate> candidates;
    for (const auto & stats : tensor_stats) {
        const auto tensor_it = cache->tensor_map.find(stats.tensor);
        if (tensor_it == cache->tensor_map.end()) {
            continue;
        }
        if (cache->prefill_markov && stats.transitions) {
            tensor_it->second->markov->seed(stats.transitions);
            ++cache->prefill_markov_tensors;
        }
        for (uint32_t expert = 0; expert < stats.n_experts; ++expert) {
            if (stats.counts[expert] > 0) {
                candidates.push_back({ tensor_it->second, expert, stats.counts[expert] });
            }
        }
    }
    ggml_metal_moe_profile_release_transitions(profile);

    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        const long double a_score = (long double) a.count / a.entry->expert_size;
        const long double b_score = (long double) b.count / b.entry->expert_size;
        return a_score != b_score ? a_score > b_score : a.count > b.count;
    });

    const size_t layer_target = cache->n_layers == 0 ? cache->capacity : cache->capacity / cache->n_layers;
    auto preload = [&](const ggml_metal_moe_preload_candidate & candidate, bool enforce_layer_target) {
        const size_t allocation_size = (candidate.entry->expert_size + 4095) / 4096 * 4096;
        if (allocation_size > cache->capacity - cache->slots.resident_bytes()) {
            return;
        }
        if (allocation_size > cache->slots.largest_free_range()) {
            return;
        }
        if (enforce_layer_target && candidate.entry->layer != UINT32_MAX &&
                allocation_size > layer_target - std::min(layer_target, cache->slots.group_resident_bytes(candidate.entry->layer))) {
            return;
        }

        const uint64_t key = ggml_metal_moe_cache_key(candidate.entry->id, candidate.expert);
        if (cache->slots.contains(key)) {
            return;
        }
        const auto result = cache->slots.prefetch(key, candidate.entry->expert_size, candidate.entry->layer);
        if (!result.admitted) {
            return;
        }
        cache->copy_expert(*candidate.entry, candidate.expert, result.offset);
        candidate.entry->page_table[candidate.expert] = result.offset;
        ++cache->preloaded_entries;
        cache->preloaded_bytes += allocation_size;
    };

    for (const auto & candidate : candidates) {
        preload(candidate, true);
    }
    for (const auto & candidate : candidates) {
        preload(candidate, false);
    }
    cache->commit_copy();
    cache->wait_for_copy();

    cache->slots.reset_stats();
    GGML_LOG_INFO("%s: entries = %zu, bytes = %zu\n", __func__, cache->preloaded_entries, cache->preloaded_bytes);
}

ggml_metal_buffer_t ggml_metal_moe_cache_get_buffer(ggml_metal_moe_cache_t cache) {
    return cache->buffer;
}

size_t ggml_metal_moe_cache_get_capacity(ggml_metal_moe_cache_t cache) {
    return cache->capacity;
}
