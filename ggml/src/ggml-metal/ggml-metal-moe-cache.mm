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
        buffer = ggml_metal_buffer_init(dev, capacity_bytes, false);
        for (size_t i = 0; i < 2; ++i) {
            staging[i] = ggml_metal_buffer_init(dev, GGML_METAL_MOE_STAGING_SIZE, true);
        }
    }

    ~ggml_metal_moe_cache() {
        wait_for_copy();
        const auto & stats = slots.stats();
        GGML_LOG_INFO("%s: accesses = %llu, hits = %llu, admission attempts = %llu, admissions = %llu, cost rejections = %llu, allocation failures = %llu, evictions = %llu, resident entries = %zu, resident bytes = %zu, staging batches = %llu, async copies = %llu, predictions = %llu, prefetch attempts = %llu, prefetch admissions = %llu, useful prefetches = %llu, wasted prefetches = %llu\n",
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
                (unsigned long long) wasted_prefetches);
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
    std::unordered_set<uint64_t> prefetched;

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
            }
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
            if (predicted.count >= 3 && (uint64_t) predicted.count * 100 >= (uint64_t) predicted.total * 8 && !cache->slots.contains(key)) {
                predictions.push_back({ &entry, predicted.state, predicted.count, predicted.total });
            }
        }
    }

    std::sort(predictions.begin(), predictions.end(), [](const prediction & a, const prediction & b) {
        const long double a_score = (long double) a.count / a.total / a.entry->expert_size;
        const long double b_score = (long double) b.count / b.total / b.entry->expert_size;
        return a_score != b_score ? a_score > b_score : a.count > b.count;
    });
    const size_t prefetch_limit = std::min<size_t>(8 * 1024 * 1024, cache->capacity / 64);
    size_t prefetch_bytes = 0;
    for (const auto & prediction : predictions) {
        const size_t allocation_size = (prediction.entry->expert_size + 4095) / 4096 * 4096;
        if (prefetch_bytes >= prefetch_limit || allocation_size > prefetch_limit - prefetch_bytes) {
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
                (uint64_t) prediction.count * 1000 / prediction.total);
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
        for (uint32_t expert = 0; expert < stats.n_experts; ++expert) {
            if (stats.counts[expert] > 0) {
                candidates.push_back({ tensor_it->second, expert, stats.counts[expert] });
            }
        }
    }

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
        cache->slots.access(key, candidate.entry->expert_size, candidate.entry->layer);
        const auto result = cache->slots.access(key, candidate.entry->expert_size, candidate.entry->layer);
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
