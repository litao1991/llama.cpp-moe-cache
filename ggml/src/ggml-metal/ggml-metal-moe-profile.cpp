#include "ggml-metal-moe-profile.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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
    uint32_t * transitions = nullptr;
    uint32_t * previous = nullptr;
    uint32_t transition_slots = 0;
    size_t transition_offset = SIZE_MAX;
    uint32_t part_mask = 0;
    bool records = true;
};

struct ggml_metal_moe_profile {
    ~ggml_metal_moe_profile() {
        if (buffer) {
            ggml_metal_buffer_free(buffer);
        }
        if (transition_buffer) {
            ggml_metal_buffer_free(transition_buffer);
        }
    }

    ggml_metal_buffer_t buffer = nullptr;
    uint32_t * counts = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    ggml_metal_buffer_t transition_buffer = nullptr;
    uint32_t * transition_counts = nullptr;
    size_t transition_capacity = 0;
    size_t transition_used = 0;
    bool active = true;
    bool logged = false;
    bool full_logged = false;
    bool transition_full_logged = false;
    bool bundle_enabled = true;
    std::mutex mutex;
    std::unordered_map<const ggml_tensor *, ggml_metal_moe_tensor_profile *> tensor_map;
    std::unordered_map<uint64_t, ggml_metal_moe_tensor_profile *> bundle_map;
    std::vector<std::unique_ptr<ggml_metal_moe_tensor_profile>> tensors;
};

static uint32_t ggml_metal_moe_profile_env_u32(const char * name, uint32_t default_value) {
    const char * value = getenv(name);
    if (!value) {
        return default_value;
    }
    char * end = nullptr;
    const unsigned long parsed = strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed <= UINT32_MAX ? (uint32_t) parsed : default_value;
}

static bool ggml_metal_moe_profile_bundle_key(const ggml_tensor * tensor, uint64_t & key, int & part) {
    const char * name = tensor->name;
    if (strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    part = strstr(name, ".ffn_gate_exps.") ? 0 : strstr(name, ".ffn_up_exps.") ? 1 : strstr(name, ".ffn_down_exps.") ? 2 : -1;
    if (part < 0) {
        return false;
    }
    char * end = nullptr;
    const unsigned long layer = strtoul(name + 4, &end, 10);
    if (end == name + 4 || *end != '.' || layer > UINT32_MAX) {
        return false;
    }
    key = ((uint64_t) layer << 32) | (uint32_t) tensor->ne[2];
    return true;
}

static struct ggml_metal_moe_profile_bindings ggml_metal_moe_profile_disabled(void) {
    return {
        false,
        false,
        { nullptr, 0 },
        { nullptr, 0 },
        { nullptr, 0 },
        nullptr,
        nullptr,
        nullptr,
        0,
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
    const char * bundle = getenv("GGML_METAL_MOE_CACHE_BUNDLE");
    profile->bundle_enabled = !bundle || strcmp(bundle, "0") != 0;

    const char * prefill_markov = getenv("GGML_METAL_MOE_CACHE_PREFILL_MARKOV");
    if (prefill_markov && strcmp(prefill_markov, "0") == 0) {
        return profile.release();
    }

    const size_t transition_capacity = std::clamp<size_t>(
            (size_t) ggml_metal_moe_profile_env_u32("GGML_METAL_MOE_PROFILE_TRANSITION_MIB", 4) * 1024 * 1024,
            1024 * 1024,
            8 * 1024 * 1024);
    profile->transition_buffer = ggml_metal_buffer_init(dev, transition_capacity, true);
    if (profile->transition_buffer && ggml_metal_buffer_is_shared(profile->transition_buffer)) {
        profile->transition_counts = (uint32_t *) ggml_metal_buffer_get_base(profile->transition_buffer);
        profile->transition_capacity = transition_capacity / sizeof(uint32_t);
    } else if (profile->transition_buffer) {
        ggml_metal_buffer_free(profile->transition_buffer);
        profile->transition_buffer = nullptr;
    }
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
    for (const auto & entry : profile->tensors) {
        if (!entry->records || !entry->transitions) {
            continue;
        }
        std::fill(entry->transitions, entry->transitions + (size_t) entry->n_experts * entry->transition_slots, 0);
        std::fill(entry->previous, entry->previous + entry->n_experts, UINT32_MAX);
    }
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
    size_t logical_profiles = 0;
    for (const auto & entry : profile->tensors) {
        if (!entry->records) {
            continue;
        }
        ++logical_profiles;
        for (uint32_t expert = 0; expert < entry->n_experts; ++expert) {
            total += entry->counts[expert];
        }
    }

    GGML_LOG_INFO("%s: tensors = %zu, logical profiles = %zu, expert selections = %llu\n",
            __func__, profile->tensors.size(), logical_profiles, (unsigned long long) total);

    for (const auto & entry : profile->tensors) {
        if (!entry->records) {
            continue;
        }
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

void ggml_metal_moe_profile_release_transitions(ggml_metal_moe_profile_t profile) {
    if (!profile) {
        return;
    }

    std::lock_guard<std::mutex> lock(profile->mutex);
    for (const auto & entry : profile->tensors) {
        entry->transitions = nullptr;
        entry->previous = nullptr;
        entry->transition_slots = 0;
        entry->transition_offset = SIZE_MAX;
    }
    if (profile->transition_buffer) {
        ggml_metal_buffer_free(profile->transition_buffer);
        profile->transition_buffer = nullptr;
    }
    profile->transition_counts = nullptr;
    profile->transition_capacity = 0;
    profile->transition_used = 0;
}

struct ggml_metal_moe_profile_stats ggml_metal_moe_profile_get_stats(ggml_metal_moe_profile_t profile) {
    if (!profile) {
        return { 0, 0 };
    }

    std::lock_guard<std::mutex> lock(profile->mutex);
    uint64_t selections = 0;
    for (const auto & entry : profile->tensors) {
        if (!entry->records) {
            continue;
        }
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
    *stats = { entry->tensor, entry->counts, entry->transitions, entry->n_experts, entry->transition_slots };
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
        ggml_metal_moe_tensor_profile * entry = existing->second;
        ggml_metal_buffer_id counts = ggml_metal_buffer_get_base_id(profile->buffer);
        counts.offs += entry->offset * sizeof(uint32_t);
        ggml_metal_buffer_id transitions = { nullptr, 0 };
        ggml_metal_buffer_id previous = { nullptr, 0 };
        if (entry->transitions) {
            transitions = ggml_metal_buffer_get_base_id(profile->transition_buffer);
            transitions.offs += entry->transition_offset * sizeof(uint32_t);
            previous = transitions;
            previous.offs += (size_t) entry->n_experts * entry->transition_slots * sizeof(uint32_t);
        }
        return {
            entry->records,
            entry->records && entry->transitions != nullptr,
            counts,
            transitions,
            previous,
            entry->counts,
            entry->transitions,
            entry->previous,
            entry->transition_slots,
        };
    }

    uint64_t bundle_key = 0;
    int bundle_part = -1;
    ggml_metal_moe_tensor_profile * bundle_owner = nullptr;
    if (profile->bundle_enabled && ggml_metal_moe_profile_bundle_key(tensor, bundle_key, bundle_part)) {
        const auto owner = profile->bundle_map.find(bundle_key);
        if (owner != profile->bundle_map.end() && (owner->second->part_mask & (1u << bundle_part)) == 0) {
            bundle_owner = owner->second;
        }
    }

    if (bundle_owner) {
        std::unique_ptr<ggml_metal_moe_tensor_profile> entry(new (std::nothrow) ggml_metal_moe_tensor_profile);
        if (!entry) {
            return ggml_metal_moe_profile_disabled();
        }
        entry->tensor = tensor;
        entry->name = tensor->name;
        entry->n_experts = bundle_owner->n_experts;
        entry->counts = bundle_owner->counts;
        entry->offset = bundle_owner->offset;
        entry->transitions = bundle_owner->transitions;
        entry->previous = bundle_owner->previous;
        entry->transition_slots = bundle_owner->transition_slots;
        entry->transition_offset = bundle_owner->transition_offset;
        entry->records = false;
        bundle_owner->part_mask |= 1u << bundle_part;

        ggml_metal_moe_tensor_profile * raw = entry.get();
        profile->tensor_map.emplace(tensor, raw);
        profile->tensors.emplace_back(std::move(entry));

        ggml_metal_buffer_id counts = ggml_metal_buffer_get_base_id(profile->buffer);
        counts.offs += raw->offset * sizeof(uint32_t);
        ggml_metal_buffer_id transitions = { nullptr, 0 };
        ggml_metal_buffer_id previous = { nullptr, 0 };
        if (raw->transitions) {
            transitions = ggml_metal_buffer_get_base_id(profile->transition_buffer);
            transitions.offs += raw->transition_offset * sizeof(uint32_t);
            previous = transitions;
            previous.offs += (size_t) raw->n_experts * raw->transition_slots * sizeof(uint32_t);
        }
        return {
            false,
            false,
            counts,
            transitions,
            previous,
            raw->counts,
            raw->transitions,
            raw->previous,
            raw->transition_slots,
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

    const uint32_t transition_slots = std::min<uint32_t>(
            ggml_metal_moe_profile_env_u32("GGML_METAL_MOE_PROFILE_TRANSITION_SLOTS", 16),
            (uint32_t) std::min<size_t>(n_experts, UINT16_MAX));
    const size_t matrix_entries = n_experts <= UINT16_MAX ? n_experts * transition_slots : 0;
    const size_t transition_entries = matrix_entries + n_experts;
    if (profile->transition_counts && n_experts <= UINT16_MAX &&
            transition_slots > 0 && transition_entries <= profile->transition_capacity - profile->transition_used) {
        entry->transition_slots = transition_slots;
        entry->transition_offset = profile->transition_used;
        entry->transitions = profile->transition_counts + entry->transition_offset;
        entry->previous = entry->transitions + matrix_entries;
        std::fill(entry->transitions, entry->transitions + matrix_entries, 0);
        std::fill(entry->previous, entry->previous + n_experts, UINT32_MAX);
        profile->transition_used += transition_entries;
    } else if (!profile->transition_full_logged) {
        GGML_LOG_WARN("%s: transition arena is full at %zu entries\n", __func__, profile->transition_used);
        profile->transition_full_logged = true;
    }

    ggml_metal_moe_tensor_profile * raw = entry.get();
    profile->tensor_map.emplace(tensor, raw);
    profile->tensors.emplace_back(std::move(entry));
    if (profile->bundle_enabled && ggml_metal_moe_profile_bundle_key(tensor, bundle_key, bundle_part)) {
        raw->part_mask = 1u << bundle_part;
        profile->bundle_map.emplace(bundle_key, raw);
    }

    ggml_metal_buffer_id counts = ggml_metal_buffer_get_base_id(profile->buffer);
    counts.offs += raw->offset * sizeof(uint32_t);
    ggml_metal_buffer_id transitions = { nullptr, 0 };
    ggml_metal_buffer_id previous = { nullptr, 0 };
    if (raw->transitions) {
        transitions = ggml_metal_buffer_get_base_id(profile->transition_buffer);
        transitions.offs += raw->transition_offset * sizeof(uint32_t);
        previous = transitions;
        previous.offs += (size_t) raw->n_experts * raw->transition_slots * sizeof(uint32_t);
    }
    return {
        true,
        raw->transitions != nullptr,
        counts,
        transitions,
        previous,
        raw->counts,
        raw->transitions,
        raw->previous,
        raw->transition_slots,
    };
}
