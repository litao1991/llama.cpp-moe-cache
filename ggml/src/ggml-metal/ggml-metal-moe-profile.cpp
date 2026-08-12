#include "ggml-metal-moe-profile.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_metal_moe_tensor_profile {
    const ggml_tensor * tensor = nullptr;
    std::string name;
    uint32_t n_experts = 0;
    uint32_t * counts = nullptr;
    size_t offset = 0;
};

struct ggml_metal_moe_profile {
    ~ggml_metal_moe_profile() {
        if (buffer) {
            ggml_metal_buffer_free(buffer);
        }
    }

    ggml_metal_buffer_t buffer = nullptr;
    uint32_t * counts = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    bool active = true;
    bool logged = false;
    bool full_logged = false;
    std::mutex mutex;
    std::unordered_map<const ggml_tensor *, ggml_metal_moe_tensor_profile *> tensor_map;
    std::vector<std::unique_ptr<ggml_metal_moe_tensor_profile>> tensors;
};

static struct ggml_metal_moe_profile_bindings ggml_metal_moe_profile_disabled(void) {
    return {
        false,
        { nullptr, 0 },
        nullptr,
    };
}

ggml_metal_moe_profile_t ggml_metal_moe_profile_init(ggml_metal_device_t dev) {
    static constexpr size_t capacity = 1024 * 1024;

    std::unique_ptr<ggml_metal_moe_profile> profile(new (std::nothrow) ggml_metal_moe_profile);
    if (!profile) {
        return nullptr;
    }

    profile->buffer = ggml_metal_buffer_init(dev, capacity, true);
    if (!profile->buffer || !ggml_metal_buffer_is_shared(profile->buffer)) {
        return nullptr;
    }

    profile->counts = (uint32_t *) ggml_metal_buffer_get_base(profile->buffer);
    profile->capacity = capacity / sizeof(uint32_t);
    std::fill(profile->counts, profile->counts + profile->capacity, 0);
    return profile.release();
}

void ggml_metal_moe_profile_free(ggml_metal_moe_profile_t profile) {
    delete profile;
}

void ggml_metal_moe_profile_reset(ggml_metal_moe_profile_t profile) {
    if (!profile) {
        return;
    }

    std::lock_guard<std::mutex> lock(profile->mutex);
    if (!profile->active) {
        return;
    }

    std::fill(profile->counts, profile->counts + profile->used, 0);
}

void ggml_metal_moe_profile_finish(ggml_metal_moe_profile_t profile) {
    if (!profile) {
        return;
    }

    std::lock_guard<std::mutex> lock(profile->mutex);
    profile->active = false;
    if (profile->logged) {
        return;
    }
    profile->logged = true;

    uint64_t total = 0;
    for (const auto & entry : profile->tensors) {
        for (uint32_t expert = 0; expert < entry->n_experts; ++expert) {
            total += entry->counts[expert];
        }
    }

    GGML_LOG_INFO("%s: tensors = %zu, expert selections = %llu\n",
            __func__, profile->tensors.size(), (unsigned long long) total);

    for (const auto & entry : profile->tensors) {
        uint64_t selections = 0;
        uint32_t active = 0;
        std::vector<std::pair<uint32_t, uint32_t>> ranked;
        ranked.reserve(entry->n_experts);

        for (uint32_t expert = 0; expert < entry->n_experts; ++expert) {
            const uint32_t count = entry->counts[expert];
            selections += count;
            active += count > 0;
            ranked.emplace_back(count, expert);
        }

        std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });

        std::string top;
        const size_t n_top = std::min<size_t>(8, ranked.size());
        for (size_t i = 0; i < n_top && ranked[i].first > 0; ++i) {
            if (!top.empty()) {
                top += ',';
            }
            top += std::to_string(ranked[i].second) + ':' + std::to_string(ranked[i].first);
        }

        GGML_LOG_INFO("ggml_metal_moe_profile: tensor = %s, experts = %u, selections = %llu, active = %u, top = %s\n",
                entry->name.c_str(), entry->n_experts, (unsigned long long) selections, active, top.c_str());
    }
}

struct ggml_metal_moe_profile_stats ggml_metal_moe_profile_get_stats(ggml_metal_moe_profile_t profile) {
    if (!profile) {
        return { 0, 0 };
    }

    std::lock_guard<std::mutex> lock(profile->mutex);
    uint64_t selections = 0;
    for (const auto & entry : profile->tensors) {
        for (uint32_t expert = 0; expert < entry->n_experts; ++expert) {
            selections += entry->counts[expert];
        }
    }
    return { profile->tensors.size(), selections };
}

bool ggml_metal_moe_profile_get_tensor_stats(
        ggml_metal_moe_profile_t profile,
        size_t index,
        struct ggml_metal_moe_tensor_profile_stats * stats) {
    if (!profile || !stats) {
        return false;
    }

    std::lock_guard<std::mutex> lock(profile->mutex);
    if (index >= profile->tensors.size()) {
        return false;
    }
    const auto & entry = profile->tensors[index];
    *stats = { entry->tensor, entry->counts, entry->n_experts };
    return true;
}

struct ggml_metal_moe_profile_bindings ggml_metal_moe_profile_get_bindings(
        ggml_metal_moe_profile_t profile,
        const struct ggml_tensor * tensor) {
    if (!profile || !tensor || tensor->ne[2] <= 1 || tensor->ne[2] > std::numeric_limits<uint32_t>::max()) {
        return ggml_metal_moe_profile_disabled();
    }

    std::lock_guard<std::mutex> lock(profile->mutex);
    if (!profile->active) {
        return ggml_metal_moe_profile_disabled();
    }

    const auto existing = profile->tensor_map.find(tensor);
    if (existing != profile->tensor_map.end()) {
        ggml_metal_buffer_id counts = ggml_metal_buffer_get_base_id(profile->buffer);
        counts.offs += existing->second->offset * sizeof(uint32_t);
        return {
            true,
            counts,
            existing->second->counts,
        };
    }

    const size_t n_experts = (size_t) tensor->ne[2];
    if (n_experts > profile->capacity - profile->used) {
        if (!profile->full_logged) {
            GGML_LOG_WARN("%s: counter arena is full at %zu entries\n", __func__, profile->used);
            profile->full_logged = true;
        }
        return ggml_metal_moe_profile_disabled();
    }

    std::unique_ptr<ggml_metal_moe_tensor_profile> entry(new (std::nothrow) ggml_metal_moe_tensor_profile);
    if (!entry) {
        return ggml_metal_moe_profile_disabled();
    }

    entry->tensor = tensor;
    entry->name = tensor->name;
    entry->n_experts = (uint32_t) tensor->ne[2];
    entry->offset = profile->used;
    entry->counts = profile->counts + entry->offset;
    profile->used += entry->n_experts;

    ggml_metal_moe_tensor_profile * raw = entry.get();
    profile->tensor_map.emplace(tensor, raw);
    profile->tensors.emplace_back(std::move(entry));

    ggml_metal_buffer_id counts = ggml_metal_buffer_get_base_id(profile->buffer);
    counts.offs += raw->offset * sizeof(uint32_t);
    return {
        true,
        counts,
        raw->counts,
    };
}
