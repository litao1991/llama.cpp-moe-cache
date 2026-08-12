#include "ggml-metal-moe-cache.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-moe-cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

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
    uint32_t n_experts = 0;
    size_t expert_size = 0;
    ggml_metal_buffer_t page_table_buffer = nullptr;
    ggml_metal_buffer_t access_buffer = nullptr;
    uint64_t * page_table = nullptr;
    uint32_t * access = nullptr;
};

struct ggml_metal_moe_cache {
    ggml_metal_moe_cache(ggml_metal_device_t dev, size_t capacity_bytes) :
        dev(dev),
        capacity(capacity_bytes),
        slots(capacity_bytes, 4096) {
        buffer = ggml_metal_buffer_init(dev, capacity_bytes, true);
    }

    ~ggml_metal_moe_cache() {
        const auto & stats = slots.stats();
        GGML_LOG_INFO("%s: accesses = %llu, hits = %llu, admissions = %llu, evictions = %llu\n",
                __func__,
                (unsigned long long) stats.accesses,
                (unsigned long long) stats.hits,
                (unsigned long long) stats.admissions,
                (unsigned long long) stats.evictions);
        tensors.clear();
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
};

static uint64_t ggml_metal_moe_cache_key(uint32_t tensor_id, uint32_t expert) {
    return ((uint64_t) tensor_id << 32) | expert;
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
    if (!cache->buffer || !ggml_metal_buffer_is_shared(cache->buffer)) {
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
    char * weights = (char *) ggml_metal_buffer_get_base(cache->buffer);

    for (const auto & entry_ptr : cache->tensors) {
        ggml_metal_moe_tensor_cache & entry = *entry_ptr;
        for (uint32_t expert = 0; expert < entry.n_experts; ++expert) {
            if (entry.access[expert] == 0) {
                continue;
            }
            entry.access[expert] = 0;

            const uint64_t key = ggml_metal_moe_cache_key(entry.id, expert);
            const auto result = cache->slots.access(key, entry.expert_size);
            for (uint64_t evicted : result.evicted) {
                const uint32_t tensor_id = (uint32_t) (evicted >> 32);
                const uint32_t expert_id = (uint32_t) evicted;
                if (tensor_id < cache->tensors.size() && expert_id < cache->tensors[tensor_id]->n_experts) {
                    cache->tensors[tensor_id]->page_table[expert_id] = UINT64_MAX;
                }
            }

            if (result.admitted) {
                memcpy(weights + result.offset, (const char *) entry.tensor->data + expert * entry.expert_size, entry.expert_size);
                entry.page_table[expert] = result.offset;
            }
        }
    }
}

ggml_metal_buffer_t ggml_metal_moe_cache_get_buffer(ggml_metal_moe_cache_t cache) {
    return cache->buffer;
}

size_t ggml_metal_moe_cache_get_capacity(ggml_metal_moe_cache_t cache) {
    return cache->capacity;
}
