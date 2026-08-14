#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_moe_profile * ggml_metal_moe_profile_t;

struct ggml_metal_moe_profile_bindings {
    bool enabled;
    bool transitions_enabled;
    struct ggml_metal_buffer_id counts;
    struct ggml_metal_buffer_id transitions;
    struct ggml_metal_buffer_id previous;
    uint32_t * host_counts;
    uint32_t * host_transitions;
    uint32_t * host_previous;
    uint32_t transition_slots;
};

struct ggml_metal_moe_profile_stats {
    size_t tensors;
    uint64_t selections;
};

struct ggml_metal_moe_tensor_profile_stats {
    const struct ggml_tensor * tensor;
    const uint32_t * counts;
    const uint32_t * transitions;
    uint32_t n_experts;
    uint32_t transition_slots;
};

ggml_metal_moe_profile_t ggml_metal_moe_profile_init(ggml_metal_device_t dev);
void ggml_metal_moe_profile_free(ggml_metal_moe_profile_t profile);

void ggml_metal_moe_profile_reset(ggml_metal_moe_profile_t profile);
void ggml_metal_moe_profile_finish(ggml_metal_moe_profile_t profile);
void ggml_metal_moe_profile_release_transitions(ggml_metal_moe_profile_t profile);
struct ggml_metal_moe_profile_stats ggml_metal_moe_profile_get_stats(ggml_metal_moe_profile_t profile);
bool ggml_metal_moe_profile_get_tensor_stats(
        ggml_metal_moe_profile_t profile,
        size_t index,
        struct ggml_metal_moe_tensor_profile_stats * stats);
struct ggml_metal_moe_profile_bindings ggml_metal_moe_profile_get_bindings(
        ggml_metal_moe_profile_t profile,
        const struct ggml_tensor * tensor);
#ifdef __cplusplus
}
#endif
