#include "ggml-metal-moe-cache.h"
#include "ggml-metal-moe-profile.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-moe-cache.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <libproc.h>
#include <sys/resource.h>
#endif
#endif

static bool ggml_metal_moe_disk_read_bytes(uint64_t & bytes) {
#if defined(__APPLE__) && TARGET_OS_OSX
    rusage_info_v2 info = {};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t *) &info) == 0) {
        bytes = info.ri_diskio_bytesread;
        return true;
    }
#endif
    (void) bytes;
    return false;
}

static uint64_t ggml_metal_moe_elapsed_ns(std::chrono::steady_clock::time_point start) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

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

static size_t ggml_metal_moe_align(size_t size_bytes) {
    return (size_bytes + 4095) / 4096 * 4096;
}

struct ggml_metal_moe_bundle_cache;

struct ggml_metal_moe_tensor_cache {
    ~ggml_metal_moe_tensor_cache() {
        if (page_table_buffer) {
            ggml_metal_buffer_free(page_table_buffer);
        }
    }

    const ggml_tensor * tensor = nullptr;
    ggml_metal_moe_bundle_cache * bundle = nullptr;
    size_t expert_size = 0;
    size_t bundle_offset = 0;
    ggml_metal_moe_expert_source file_source = { -1, 0, 0, 0 };
    ggml_metal_buffer_t page_table_buffer = nullptr;
    uint64_t * page_table = nullptr;
};

struct ggml_metal_moe_bundle_cache {
    ~ggml_metal_moe_bundle_cache() {
        if (access_buffer) {
            ggml_metal_buffer_free(access_buffer);
        }
    }

    uint32_t id = 0;
    uint32_t layer = UINT32_MAX;
    uint32_t n_experts = 0;
    uint32_t part_mask = 0;
    uint64_t eval_step = UINT64_MAX;
    std::vector<uint8_t> prefill_top;
    size_t expert_size = 0;
    ggml_metal_buffer_t access_buffer = nullptr;
    uint32_t * access = nullptr;
    std::unique_ptr<ggml_moe_markov> markov;
    std::vector<ggml_metal_moe_tensor_cache *> members;
};

struct ggml_metal_moe_cache {
    ggml_metal_moe_cache(ggml_metal_device_t dev, size_t capacity_bytes) :
        dev(dev),
        capacity(capacity_bytes),
        slots(capacity_bytes, 4096, std::min(100u, ggml_metal_moe_env_u32("GGML_METAL_MOE_CACHE_PROTECTED_PERCENT", 80))) {
        prefetch_enabled = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_PREFETCH", true);
        adaptive_prefetch = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_ADAPTIVE_PREFETCH", true);
        prefill_markov = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_PREFILL_MARKOV", true);
        bundle_enabled = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_BUNDLE", true);
        dynamic_layer_budget = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_DYNAMIC_LAYER_BUDGET", true);
        remaining_tokens_hint = ggml_metal_moe_env_u32("GGML_METAL_MOE_CACHE_REMAINING_TOKENS", UINT32_MAX);
        eval_enabled = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_EVAL", false);
        strict_update = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_STRICT_UPDATE", true);
        strict_mm = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_MM_ID", true);
        preload_enabled = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_PRELOAD", true);
        preload_rebuild = ggml_metal_moe_env_enabled("GGML_METAL_MOE_CACHE_PRELOAD_REBUILD", false);
        protected_percent = std::min(100u, ggml_metal_moe_env_u32("GGML_METAL_MOE_CACHE_PROTECTED_PERCENT", 80));
        buffer = ggml_metal_buffer_init(dev, capacity_bytes, true);
    }

    ~ggml_metal_moe_cache() {
        wasted_prefetches += prefetched.size();
        for (size_t i = 0; i < prefetched.size(); ++i) {
            prefetch_control.record_outcome(false);
        }
        prefetched.clear();
        const auto stats = slots.stats();
        GGML_LOG_INFO("%s: accesses = %llu, hits = %llu, admission attempts = %llu, admissions = %llu, cost rejections = %llu, allocation failures = %llu, evictions = %llu, resident entries = %zu, resident bytes = %zu, staging batches = %llu, async copies = %llu, synchronous loads = %llu, synchronous hits = %llu, predictions = %llu, prefetch attempts = %llu, prefetch admissions = %llu, useful prefetches = %llu, wasted prefetches = %llu, adaptive skips = %llu, copy pressure updates = %llu, remaining stops = %llu, controller outcomes = %u/%u, controller updates = %u, prefill Markov bundles = %zu, tensors = %zu, bundles = %zu, direct bytes = %llu, staged bytes = %llu\n",
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
                (unsigned long long) synchronous_loads,
                (unsigned long long) synchronous_hits,
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
                prefill_markov_tensors,
                tensors.size(),
                bundles.size(),
                (unsigned long long) direct_copy_bytes,
                (unsigned long long) staged_copy_bytes);
        GGML_LOG_INFO("%s: shared cache, file read bytes = %llu, host copy bytes = %llu, read failures = %llu\n", __func__,
                (unsigned long long) file_read_bytes,
                (unsigned long long) host_copy_bytes,
                (unsigned long long) read_failures);
        if (eval_enabled) {
            GGML_LOG_INFO("moe_cache_iteration: require calls = %llu, all hit calls = %llu, wait calls = %llu, wait ns = %llu, gpu ns = %llu, prefill top hits = %llu, prefill early hits = %llu, early required = %llu\n",
                    (unsigned long long) eval_require_calls, (unsigned long long) eval_all_hit_calls,
                    (unsigned long long) eval_wait_calls, (unsigned long long) eval_wait_ns,
                    (unsigned long long) eval_gpu_ns, (unsigned long long) eval_prefill_top_hits,
                    (unsigned long long) eval_prefill_early_hits, (unsigned long long) eval_early_required);
            GGML_LOG_INFO("moe_cache_eval: steps = %llu, route calls = %llu, required = %llu, resident hits = %llu, sync bundles = %llu, sync bytes = %llu, decode read bytes = %llu, decode read calls = %llu, decode read ns = %llu, require ns = %llu, require read ns = %llu, disk bytes = %llu, disk available = %u, capacity = %zu\n",
                    (unsigned long long) eval_steps, (unsigned long long) eval_route_calls,
                    (unsigned long long) eval_required, (unsigned long long) eval_hits,
                    (unsigned long long) eval_sync_bundles, (unsigned long long) eval_sync_bytes,
                    (unsigned long long) eval_read_bytes, (unsigned long long) eval_read_calls,
                    (unsigned long long) eval_read_ns, (unsigned long long) eval_require_ns,
                    (unsigned long long) eval_require_read_ns,
                    (unsigned long long) (eval_disk_end >= eval_disk_start ? eval_disk_end - eval_disk_start : 0),
                    eval_disk_available ? 1u : 0u, capacity);
        }
        tensors.clear();
        bundles.clear();
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
    std::unordered_map<const ggml_tensor *, ggml_metal_moe_expert_source> file_sources;
    std::vector<std::unique_ptr<ggml_metal_moe_tensor_cache>> tensors;
    std::unordered_map<uint64_t, ggml_metal_moe_bundle_cache *> bundle_map;
    std::vector<std::unique_ptr<ggml_metal_moe_bundle_cache>> bundles;
    uint32_t n_layers = 0;
    size_t preloaded_entries = 0;
    size_t preloaded_bytes = 0;
    uint64_t staging_batches = 0;
    uint64_t async_copies = 0;
    uint64_t direct_copy_bytes = 0;
    uint64_t staged_copy_bytes = 0;
    uint64_t file_read_bytes = 0;
    uint64_t host_copy_bytes = 0;
    uint64_t read_failures = 0;
    bool eval_enabled = false;
    bool eval_active = false;
    bool eval_pending = false;
    bool eval_disk_available = false;
    uint64_t eval_steps = 0;
    uint64_t eval_route_calls = 0;
    uint64_t eval_required = 0;
    uint64_t eval_hits = 0;
    uint64_t eval_sync_bundles = 0;
    uint64_t eval_sync_bytes = 0;
    uint64_t eval_read_bytes = 0;
    uint64_t eval_read_calls = 0;
    uint64_t eval_read_ns = 0;
    uint64_t eval_require_ns = 0;
    uint64_t eval_require_read_ns = 0;
    uint64_t eval_disk_start = 0;
    uint64_t eval_disk_end = 0;
    uint64_t eval_require_calls = 0;
    uint64_t eval_all_hit_calls = 0;
    uint64_t eval_wait_calls = 0;
    uint64_t eval_wait_ns = 0;
    uint64_t eval_gpu_ns = 0;
    uint64_t eval_prefill_top_hits = 0;
    uint64_t eval_prefill_early_hits = 0;
    uint64_t eval_early_required = 0;
    bool decode_phase = false;
    bool strict_update = true;
    bool strict_mm = true;
    bool preload_enabled = true;
    bool preload_rebuild = false;
    uint32_t protected_percent = 80;
    uint64_t synchronous_loads = 0;
    uint64_t synchronous_hits = 0;
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
    bool bundle_enabled = true;
    bool dynamic_layer_budget = true;
    bool preload_complete = false;
    uint32_t remaining_tokens_hint = UINT32_MAX;
    ggml_moe_prefetch_control prefetch_control;
    std::unordered_set<uint64_t> prefetched;

    bool copy_part(const ggml_metal_moe_tensor_cache & entry, uint32_t expert, size_t offset) {
        GGML_ASSERT(buffer && ggml_metal_buffer_is_shared(buffer));
        GGML_ASSERT(offset <= capacity && entry.expert_size <= capacity - offset);
        char * destination = (char *) ggml_metal_buffer_get_base(buffer) + offset;
        if (entry.file_source.fd < 0) {
            memcpy(destination, (const char *) entry.tensor->data + expert * entry.expert_size, entry.expert_size);
            host_copy_bytes += entry.expert_size;
            return true;
        }

        size_t copied = 0;
        while (copied < entry.expert_size) {
            const uint64_t file_offset = entry.file_source.file_offset + (uint64_t) expert * entry.expert_size + copied;
            const bool measure = eval_enabled && eval_active;
            const auto started = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const ssize_t read_bytes = pread(entry.file_source.fd, destination + copied, entry.expert_size - copied, (off_t) file_offset);
            if (measure) {
                eval_read_ns += ggml_metal_moe_elapsed_ns(started);
                ++eval_read_calls;
                eval_read_bytes += read_bytes > 0 ? (uint64_t) read_bytes : 0;
            }
            if (read_bytes < 0 && errno == EINTR) {
                continue;
            }
            if (read_bytes <= 0) {
                ++read_failures;
                GGML_LOG_ERROR("%s: failed to read expert %u from '%s' at offset %llu\n", __func__, expert, entry.tensor->name, (unsigned long long) file_offset);
                return false;
            }
            copied += (size_t) read_bytes;
            file_read_bytes += (size_t) read_bytes;
        }
        return true;
    }

    bool copy_expert(const ggml_metal_moe_bundle_cache & bundle, uint32_t expert, size_t offset) {
        for (const auto * member : bundle.members) {
            if (!copy_part(*member, expert, offset + member->bundle_offset)) {
                return false;
            }
        }
        return true;
    }

    void set_page_tables(const ggml_metal_moe_bundle_cache & bundle, uint32_t expert, size_t offset) {
        for (const auto * member : bundle.members) {
            member->page_table[expert] = offset + member->bundle_offset;
        }
    }

    void invalidate(uint64_t key) {
        if (prefetched.erase(key) > 0) {
            ++wasted_prefetches;
            prefetch_control.record_outcome(false);
        }
        const uint32_t bundle_id = (uint32_t) (key >> 32);
        const uint32_t expert_id = (uint32_t) key;
        if (bundle_id < bundles.size() && expert_id < bundles[bundle_id]->n_experts) {
            for (const auto * member : bundles[bundle_id]->members) {
                member->page_table[expert_id] = UINT64_MAX;
            }
        }
    }

    bool resize(size_t capacity_bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        struct resident {
            uint64_t key;
            uint64_t score;
        };
        std::vector<resident> residents;
        residents.reserve(slots.resident_entries());
        for (const auto & bundle : bundles) {
            for (uint32_t expert = 0; expert < bundle->n_experts; ++expert) {
                const uint64_t key = ((uint64_t) bundle->id << 32) | expert;
                if (slots.contains(key)) {
                    residents.push_back({
                        key,
                        (uint64_t) slots.short_frequency(key) * 4 + slots.frequency(key),
                    });
                }
                for (const auto * member : bundle->members) {
                    member->page_table[expert] = UINT64_MAX;
                }
            }
        }
        std::sort(residents.begin(), residents.end(), [](const resident & a, const resident & b) {
            return a.score != b.score ? a.score > b.score : a.key < b.key;
        });

        if (buffer) {
            ggml_metal_buffer_free(buffer);
            buffer = nullptr;
        }
        capacity = capacity_bytes;
        slots.reset(capacity_bytes, 4096, protected_percent);
        if (n_layers > 0) {
            slots.set_group_count(n_layers);
        }
        prefetched.clear();
        if (capacity_bytes == 0) {
            return true;
        }

        buffer = ggml_metal_buffer_init(dev, capacity_bytes, true);
        if (!buffer || !ggml_metal_buffer_is_shared(buffer)) {
            if (buffer) {
                ggml_metal_buffer_free(buffer);
                buffer = nullptr;
            }
            capacity = 0;
            slots.reset(0, 4096, protected_percent);
            return false;
        }
        for (const auto & resident : residents) {
            const uint32_t bundle_id = (uint32_t) (resident.key >> 32);
            const uint32_t expert = (uint32_t) resident.key;
            auto & bundle = *bundles[bundle_id];
            const auto result = slots.prefetch(resident.key, bundle.expert_size, bundle.layer, resident.score * 1024);
            if (!result.admitted) {
                continue;
            }
            if (!copy_expert(bundle, expert, result.offset)) {
                slots.erase(resident.key);
                invalidate(resident.key);
                return false;
            }
            set_page_tables(bundle, expert, result.offset);
        }
        return true;
    }
};

struct ggml_metal_moe_preload_candidate {
    ggml_metal_moe_bundle_cache * bundle;
    uint32_t expert;
    uint32_t count;
};

static uint64_t ggml_metal_moe_cache_key(uint32_t bundle_id, uint32_t expert) {
    return ((uint64_t) bundle_id << 32) | expert;
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

static int ggml_metal_moe_cache_part(const char * name) {
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    if (strstr(name, ".ffn_gate_exps.")) {
        return 0;
    }
    if (strstr(name, ".ffn_up_exps.")) {
        return 1;
    }
    if (strstr(name, ".ffn_down_exps.")) {
        return 2;
    }
    return -1;
}

static uint64_t ggml_metal_moe_cache_bundle_key(uint32_t layer, uint32_t n_experts) {
    return ((uint64_t) layer << 32) | n_experts;
}

static struct ggml_metal_moe_cache_bindings ggml_metal_moe_cache_disabled(void) {
    return {
        false,
        false,
        false,
        { nullptr, 0 },
        { nullptr, 0 },
        { nullptr, 0 },
    };
}

static bool ggml_metal_moe_bundle_has_file_sources(const ggml_metal_moe_bundle_cache & bundle) {
    return !bundle.members.empty() && std::all_of(bundle.members.begin(), bundle.members.end(), [](const auto * member) {
        return member->file_source.fd >= 0;
    });
}

ggml_metal_moe_cache_t ggml_metal_moe_cache_init(ggml_metal_device_t dev, size_t capacity_bytes) {
    std::unique_ptr<ggml_metal_moe_cache> cache(new ggml_metal_moe_cache(dev, capacity_bytes));
    if (!cache->buffer || !ggml_metal_buffer_is_shared(cache->buffer)) {
        return nullptr;
    }
    GGML_LOG_INFO("%s: shared expert cache = %.2f MiB, file reads write directly to cache slots\n", __func__, capacity_bytes / 1024.0 / 1024.0);
    return cache.release();
}

void ggml_metal_moe_cache_free(ggml_metal_moe_cache_t cache) {
    delete cache;
}

bool ggml_metal_moe_cache_resize(ggml_metal_moe_cache_t cache, size_t capacity_bytes) {
    return cache && cache->resize(capacity_bytes);
}

bool ggml_metal_moe_cache_set_source(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * tensor,
        struct ggml_metal_moe_expert_source source) {
    if (!cache || !tensor || source.fd < 0 || source.byte_size < ggml_nbytes(tensor)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->file_sources[tensor] = source;
        const auto entry = cache->tensor_map.find(tensor);
        if (entry != cache->tensor_map.end()) {
            entry->second->file_source = source;
        }
    }
    return ggml_metal_moe_cache_get_bindings(cache, tensor).enabled;
}

bool ggml_metal_moe_cache_needs_residency(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * tensor) {
    if (!cache || !tensor || !cache->buffer) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mutex);
    const auto entry = cache->tensor_map.find(tensor);
    return entry != cache->tensor_map.end() && ggml_metal_moe_bundle_has_file_sources(*entry->second->bundle);
}

bool ggml_metal_moe_cache_same_bundle(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * first,
        const struct ggml_tensor * second) {
    if (!cache || !first || !second) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mutex);
    const auto a = cache->tensor_map.find(first);
    const auto b = cache->tensor_map.find(second);
    return a != cache->tensor_map.end() && b != cache->tensor_map.end() && a->second->bundle == b->second->bundle;
}

void ggml_metal_moe_cache_record_wait(ggml_metal_moe_cache_t cache, uint32_t n_tokens, uint64_t ns, uint64_t gpu_ns) {
    if (cache && cache->eval_enabled && cache->decode_phase && n_tokens == 1) {
        ++cache->eval_wait_calls;
        cache->eval_wait_ns += ns;
        cache->eval_gpu_ns += gpu_ns;
    }
}

bool ggml_metal_moe_cache_require(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * tensor,
        const int32_t * ids,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        size_t token_stride) {
    if (!cache || !tensor || !ids || n_tokens == 0 || n_expert_used == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mutex);
    const auto entry_it = cache->tensor_map.find(tensor);
    if (entry_it == cache->tensor_map.end()) {
        return false;
    }
    auto & bundle = *entry_it->second->bundle;
    if (!ggml_metal_moe_bundle_has_file_sources(bundle)) {
        return false;
    }

    const bool measure = cache->eval_enabled && cache->decode_phase && n_tokens == 1;
    cache->eval_require_calls += measure;
    const auto started = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (measure && !cache->eval_active) {
        cache->eval_active = true;
        cache->eval_disk_available = ggml_metal_moe_disk_read_bytes(cache->eval_disk_start);
    }
    const uint64_t read_before = cache->eval_read_ns;
    const uint64_t bytes_before = cache->file_read_bytes;
    const bool first_route = measure && bundle.eval_step != cache->eval_steps;
    if (first_route) {
        bundle.eval_step = cache->eval_steps;
        ++cache->eval_route_calls;
        cache->eval_pending = true;
    }

    std::vector<uint64_t> required;
    for (uint32_t token = 0; token < n_tokens; ++token) {
        const auto * token_ids = (const int32_t *) ((const char *) ids + token * token_stride);
        for (uint32_t index = 0; index < n_expert_used; ++index) {
            const int32_t expert = token_ids[index];
            if (expert < 0 || expert >= (int32_t) bundle.n_experts) {
                return false;
            }
            required.push_back(ggml_metal_moe_cache_key(bundle.id, (uint32_t) expert));
        }
    }
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    if (required.size() * ggml_metal_moe_align(bundle.expert_size) > cache->capacity) {
        return false;
    }

    bool all_hit = true;
    for (uint64_t key : required) {
        const uint32_t expert = (uint32_t) key;
        if (first_route) {
            ++cache->eval_required;
            cache->eval_hits += cache->slots.contains(key);
            const bool predicted = expert < bundle.prefill_top.size() && bundle.prefill_top[expert];
            cache->eval_prefill_top_hits += predicted;
            if (cache->eval_steps < 32) {
                ++cache->eval_early_required;
                cache->eval_prefill_early_hits += predicted;
            }
        }
        const auto result = cache->slots.require(key, bundle.expert_size, bundle.layer, required);
        for (uint64_t evicted : result.evicted) {
            cache->invalidate(evicted);
        }
        if (result.hit) {
            ++cache->synchronous_hits;
            continue;
        }
        if (!result.admitted) {
            return false;
        }
        all_hit = false;
        if (!cache->copy_expert(bundle, expert, result.offset)) {
            cache->slots.erase(key);
            cache->invalidate(key);
            return false;
        }
        cache->set_page_tables(bundle, expert, result.offset);
        ++cache->synchronous_loads;
        cache->eval_sync_bundles += measure;
    }
    if (measure) {
        cache->eval_all_hit_calls += all_hit;
        cache->eval_require_ns += ggml_metal_moe_elapsed_ns(started);
        cache->eval_require_read_ns += cache->eval_read_ns - read_before;
        cache->eval_sync_bytes += cache->file_read_bytes - bytes_before;
    }
    return true;
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
    if (!cache || !cache->buffer || !tensor || tensor->ne[2] <= 1 || tensor->ne[2] > std::numeric_limits<uint32_t>::max() || tensor->nb[2] > cache->capacity) {
        return ggml_metal_moe_cache_disabled();
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    const auto existing = cache->tensor_map.find(tensor);
    if (existing != cache->tensor_map.end()) {
        const auto * entry = existing->second;
        return {
            true,
            ggml_metal_moe_bundle_has_file_sources(*entry->bundle),
            cache->strict_mm,
            ggml_metal_buffer_get_base_id(cache->buffer),
            ggml_metal_buffer_get_base_id(entry->page_table_buffer),
            ggml_metal_buffer_get_base_id(entry->bundle->access_buffer),
        };
    }

    if (cache->tensors.size() >= std::numeric_limits<uint32_t>::max()) {
        return ggml_metal_moe_cache_disabled();
    }

    const uint32_t n_experts = (uint32_t) tensor->ne[2];
    const uint32_t layer = ggml_metal_moe_cache_layer(tensor->name);
    const int part = ggml_metal_moe_cache_part(tensor->name);
    const auto file_source_it = cache->file_sources.find(tensor);
    const bool has_file_source = file_source_it != cache->file_sources.end();
    ggml_metal_buffer_id source = { nullptr, 0 };
    if (!has_file_source) {
        ggml_backend_buffer_t backend_buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!backend_buffer || ggml_backend_buffer_get_size(backend_buffer) == 0) {
            return ggml_metal_moe_cache_disabled();
        }
        ggml_metal_buffer_t source_buffer = (ggml_metal_buffer_t) backend_buffer->context;
        if (!ggml_metal_buffer_is_shared(source_buffer)) {
            return ggml_metal_moe_cache_disabled();
        }
        source = ggml_metal_buffer_get_id(source_buffer, tensor);
    }
    if (!has_file_source && !source.metal) {
        return ggml_metal_moe_cache_disabled();
    }

    ggml_metal_moe_bundle_cache * bundle = nullptr;
    uint64_t bundle_key = 0;
    if (cache->bundle_enabled && !cache->preload_complete && layer != UINT32_MAX && part >= 0) {
        bundle_key = ggml_metal_moe_cache_bundle_key(layer, n_experts);
        const auto bundle_it = cache->bundle_map.find(bundle_key);
        if (bundle_it != cache->bundle_map.end() &&
                (bundle_it->second->part_mask & (1u << part)) == 0 &&
                ggml_metal_moe_align(bundle_it->second->expert_size) + ggml_metal_moe_align(tensor->nb[2]) <= cache->capacity) {
            bundle = bundle_it->second;
        }
    }

    std::unique_ptr<ggml_metal_moe_bundle_cache> new_bundle;
    if (!bundle) {
        if (cache->bundles.size() >= std::numeric_limits<uint32_t>::max()) {
            return ggml_metal_moe_cache_disabled();
        }
        new_bundle.reset(new ggml_metal_moe_bundle_cache);
        new_bundle->id = (uint32_t) cache->bundles.size();
        new_bundle->layer = layer;
        new_bundle->n_experts = n_experts;
        new_bundle->access_buffer = ggml_metal_buffer_init(cache->dev, n_experts * sizeof(uint32_t), true);
        if (!new_bundle->access_buffer || !ggml_metal_buffer_is_shared(new_bundle->access_buffer)) {
            return ggml_metal_moe_cache_disabled();
        }
        new_bundle->access = (uint32_t *) ggml_metal_buffer_get_base(new_bundle->access_buffer);
        new_bundle->markov.reset(new ggml_moe_markov(n_experts, 8));
        std::fill(new_bundle->access, new_bundle->access + n_experts, 0);
        bundle = new_bundle.get();
    }

    std::unique_ptr<ggml_metal_moe_tensor_cache> entry(new ggml_metal_moe_tensor_cache);
    entry->tensor = tensor;
    entry->bundle = bundle;
    entry->expert_size = tensor->nb[2];
    entry->bundle_offset = ggml_metal_moe_align(bundle->expert_size);
    if (has_file_source) {
        entry->file_source = file_source_it->second;
    }
    entry->page_table_buffer = ggml_metal_buffer_init(cache->dev, n_experts * sizeof(uint64_t), true);
    if (!entry->page_table_buffer || !ggml_metal_buffer_is_shared(entry->page_table_buffer)) {
        return ggml_metal_moe_cache_disabled();
    }

    entry->page_table = (uint64_t *) ggml_metal_buffer_get_base(entry->page_table_buffer);
    std::fill(entry->page_table, entry->page_table + n_experts, UINT64_MAX);

    if (new_bundle) {
        if (cache->bundle_enabled && !cache->preload_complete && layer != UINT32_MAX && part >= 0) {
            cache->bundle_map.emplace(bundle_key, bundle);
        }
        cache->bundles.emplace_back(std::move(new_bundle));
        if (layer != UINT32_MAX) {
            cache->n_layers = std::max(cache->n_layers, layer + 1);
            cache->slots.set_group_count(cache->n_layers);
        }
    }
    if (part >= 0) {
        bundle->part_mask |= 1u << part;
    }
    bundle->expert_size = entry->bundle_offset + ggml_metal_moe_align(entry->expert_size);

    ggml_metal_moe_tensor_cache * raw = entry.get();
    bundle->members.push_back(raw);
    cache->tensor_map.emplace(tensor, raw);
    cache->tensors.emplace_back(std::move(entry));

    return {
        true,
        ggml_metal_moe_bundle_has_file_sources(*raw->bundle),
        cache->strict_mm,
        ggml_metal_buffer_get_base_id(cache->buffer),
        ggml_metal_buffer_get_base_id(raw->page_table_buffer),
        ggml_metal_buffer_get_base_id(raw->bundle->access_buffer),
    };
}

void ggml_metal_moe_cache_update(ggml_metal_moe_cache_t cache) {
    if (!cache) {
        return;
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->preload_complete = true;
    struct prediction {
        ggml_metal_moe_bundle_cache * bundle;
        uint32_t expert;
        uint32_t count;
        uint32_t total;
    };
    std::vector<prediction> predictions;
    uint32_t demand_evictions = 0;
    bool observed_routes = false;

    for (const auto & bundle_ptr : cache->bundles) {
        ggml_metal_moe_bundle_cache & bundle = *bundle_ptr;
        std::vector<uint32_t> current_experts;
        for (uint32_t expert = 0; expert < bundle.n_experts; ++expert) {
            if (bundle.access[expert] == 0) {
                continue;
            }
            bundle.access[expert] = 0;
            current_experts.push_back(expert);

            const uint64_t key = ggml_metal_moe_cache_key(bundle.id, expert);
            const bool already_loaded = cache->strict_update && ggml_metal_moe_bundle_has_file_sources(bundle);
            const auto result = cache->slots.access(key, bundle.expert_size, bundle.layer, !already_loaded);
            if (result.hit && cache->prefetched.erase(key) > 0) {
                ++cache->useful_prefetches;
                cache->prefetch_control.record_outcome(true);
            }
            demand_evictions += (uint32_t) result.evicted.size();
            for (uint64_t evicted : result.evicted) {
                cache->invalidate(evicted);
            }

            if (result.admitted) {
                if (cache->copy_expert(bundle, expert, result.offset)) {
                    cache->set_page_tables(bundle, expert, result.offset);
                } else {
                    cache->slots.erase(key);
                    cache->invalidate(key);
                }
            }
        }
        for (const auto & predicted : bundle.markov->observe(current_experts)) {
            ++cache->predictions;
            const uint64_t key = ggml_metal_moe_cache_key(bundle.id, predicted.state);
            if (predicted.count > 0 && predicted.total > 0 && !cache->slots.contains(key)) {
                predictions.push_back({ &bundle, predicted.state, predicted.count, predicted.total });
            }
        }
        observed_routes = observed_routes || !current_experts.empty();
    }

    if (!observed_routes) {
        return;
    }

    std::sort(predictions.begin(), predictions.end(), [](const prediction & a, const prediction & b) {
        const long double a_score = (long double) a.count / a.total / a.bundle->expert_size;
        const long double b_score = (long double) b.count / b.total / b.bundle->expert_size;
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
        cache->remaining_stops += remaining_tokens <= 1;
        limits = cache->prefetch_control.limits(base_prefetch_limit, false, demand_evictions, remaining_tokens);
    }
    if (limits.max_bytes == 0 && !predictions.empty()) {
        ++cache->adaptive_skips;
    }
    size_t prefetch_bytes = 0;
    for (const auto & prediction : predictions) {
        const size_t allocation_size = ggml_metal_moe_align(prediction.bundle->expert_size);
        if ((uint64_t) prediction.count * 100 < (uint64_t) prediction.total * limits.min_probability_percent ||
                prefetch_bytes >= limits.max_bytes || allocation_size > limits.max_bytes - prefetch_bytes) {
            continue;
        }
        const uint64_t key = ggml_metal_moe_cache_key(prediction.bundle->id, prediction.expert);
        if (cache->slots.contains(key)) {
            continue;
        }
        ++cache->prefetch_attempts;
        const auto result = cache->slots.prefetch(
                key,
                prediction.bundle->expert_size,
                prediction.bundle->layer,
                (uint64_t) prediction.count * 1024 / prediction.total);
        for (uint64_t evicted : result.evicted) {
            cache->invalidate(evicted);
        }
        if (result.admitted) {
            if (!cache->copy_expert(*prediction.bundle, prediction.expert, result.offset)) {
                cache->slots.erase(key);
                cache->invalidate(key);
                continue;
            }
            cache->set_page_tables(*prediction.bundle, prediction.expert, result.offset);
            cache->prefetched.insert(key);
            ++cache->prefetch_admissions;
            prefetch_bytes += allocation_size;
        }
    }
    if (cache->eval_pending) {
        ++cache->eval_steps;
        cache->eval_pending = false;
        cache->eval_disk_available = ggml_metal_moe_disk_read_bytes(cache->eval_disk_end) && cache->eval_disk_available;
    }
}

void ggml_metal_moe_cache_attach_profile(
        ggml_metal_moe_cache_t cache,
        ggml_metal_moe_profile_t profile) {
    if (!cache || !profile) {
        return;
    }
    for (size_t index = 0;; ++index) {
        ggml_metal_moe_tensor_profile_stats stats = {};
        if (!ggml_metal_moe_profile_get_tensor_stats(profile, index, &stats)) {
            break;
        }
        ggml_metal_moe_cache_get_bindings(cache, stats.tensor);
    }
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
    cache->preload_complete = true;
    std::vector<ggml_metal_moe_preload_candidate> candidates;
    std::unordered_set<uint32_t> seen_bundles;
    std::unordered_set<uint32_t> seeded_bundles;
    for (const auto & stats : tensor_stats) {
        const auto tensor_it = cache->tensor_map.find(stats.tensor);
        if (tensor_it == cache->tensor_map.end()) {
            continue;
        }
        auto * bundle = tensor_it->second->bundle;
        if (cache->prefill_markov && stats.transitions && seeded_bundles.insert(bundle->id).second) {
            bundle->markov->seed_sparse(stats.transitions, stats.transition_slots);
            ++cache->prefill_markov_tensors;
        }
        if (!seen_bundles.insert(bundle->id).second) {
            continue;
        }
        for (uint32_t expert = 0; expert < stats.n_experts; ++expert) {
            if (stats.counts[expert] > 0) {
                candidates.push_back({ bundle, expert, stats.counts[expert] });
            }
        }
    }
    ggml_metal_moe_profile_release_transitions(profile);

    if (cache->eval_enabled && cache->n_layers > 0) {
        for (auto & bundle : cache->bundles) {
            bundle->prefill_top.assign(bundle->n_experts, 0);
            std::vector<ggml_metal_moe_preload_candidate> ranked;
            for (const auto & candidate : candidates) {
                if (candidate.bundle == bundle.get()) {
                    ranked.push_back(candidate);
                }
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
                return a.count != b.count ? a.count > b.count : a.expert < b.expert;
            });
            const size_t limit = cache->capacity / cache->n_layers / ggml_metal_moe_align(bundle->expert_size);
            for (size_t i = 0; i < std::min(limit, ranked.size()); ++i) {
                bundle->prefill_top[ranked[i].expert] = 1;
            }
        }
    }

    if (cache->preload_enabled && cache->preload_rebuild) {
        cache->slots.reset(cache->capacity, 4096, cache->protected_percent);
        cache->prefetched.clear();
        for (const auto & bundle : cache->bundles) {
            for (const auto * member : bundle->members) {
                std::fill(member->page_table, member->page_table + bundle->n_experts, UINT64_MAX);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        const long double a_score = (long double) a.count / a.bundle->expert_size;
        const long double b_score = (long double) b.count / b.bundle->expert_size;
        return a_score != b_score ? a_score > b_score : a.count > b.count;
    });

    std::vector<size_t> layer_budgets(cache->n_layers, 0);
    if (cache->n_layers > 0) {
        if (cache->dynamic_layer_budget) {
            std::vector<ggml_moe_layer_candidate> layer_candidates;
            layer_candidates.reserve(candidates.size());
            for (const auto & candidate : candidates) {
                layer_candidates.push_back({
                    candidate.bundle->layer,
                    ggml_metal_moe_align(candidate.bundle->expert_size),
                    candidate.count,
                    candidate.bundle->expert_size,
                });
            }
            layer_budgets = ggml_moe_layer_budgets(cache->capacity, cache->n_layers, layer_candidates);
        } else {
            for (uint32_t layer = 0; layer < cache->n_layers; ++layer) {
                layer_budgets[layer] = cache->capacity / cache->n_layers + (layer < cache->capacity % cache->n_layers);
            }
        }
        cache->slots.set_group_budgets(layer_budgets);
    }

    auto preload = [&](const ggml_metal_moe_preload_candidate & candidate) {
        const size_t allocation_size = ggml_metal_moe_align(candidate.bundle->expert_size);
        if (allocation_size > cache->capacity - cache->slots.resident_bytes()) {
            return;
        }
        if (allocation_size > cache->slots.largest_free_range()) {
            return;
        }
        if (candidate.bundle->layer != UINT32_MAX && candidate.bundle->layer < layer_budgets.size() &&
                allocation_size > layer_budgets[candidate.bundle->layer] - std::min(layer_budgets[candidate.bundle->layer], cache->slots.group_resident_bytes(candidate.bundle->layer))) {
            return;
        }

        const uint64_t key = ggml_metal_moe_cache_key(candidate.bundle->id, candidate.expert);
        if (cache->slots.contains(key)) {
            return;
        }
        const auto result = cache->slots.prefetch(key, candidate.bundle->expert_size, candidate.bundle->layer);
        if (!result.admitted) {
            return;
        }
        if (!cache->copy_expert(*candidate.bundle, candidate.expert, result.offset)) {
            cache->slots.erase(key);
            cache->invalidate(key);
            return;
        }
        cache->set_page_tables(*candidate.bundle, candidate.expert, result.offset);
        ++cache->preloaded_entries;
        cache->preloaded_bytes += allocation_size;
    };

    if (cache->preload_enabled) {
        for (const auto & candidate : candidates) {
            preload(candidate);
        }
    }
    cache->slots.reset_stats();
    const auto budget_range = std::minmax_element(layer_budgets.begin(), layer_budgets.end());
    const size_t min_budget = layer_budgets.empty() ? 0 : *budget_range.first;
    const size_t max_budget = layer_budgets.empty() ? 0 : *budget_range.second;
    GGML_LOG_INFO("%s: entries = %zu, bytes = %zu, bundles = %zu, dynamic layer budget = %s, layer budget range = %zu-%zu\n",
            __func__, cache->preloaded_entries, cache->preloaded_bytes, cache->bundles.size(),
            cache->dynamic_layer_budget ? "on" : "off", min_budget, max_budget);
    cache->decode_phase = true;
}

ggml_metal_buffer_t ggml_metal_moe_cache_get_buffer(ggml_metal_moe_cache_t cache) {
    return cache->buffer;
}

size_t ggml_metal_moe_cache_get_capacity(ggml_metal_moe_cache_t cache) {
    return cache->capacity;
}
