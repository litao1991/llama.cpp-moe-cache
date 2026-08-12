#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_moe_profile * ggml_metal_moe_profile_t;

struct ggml_metal_moe_profile_bindings {
    bool enabled;
    struct ggml_metal_buffer_id counts;
    uint32_t * host_counts;
};

struct ggml_metal_moe_profile_stats {
    size_t tensors;
    uint64_t selections;
};

ggml_metal_moe_profile_t ggml_metal_moe_profile_init(ggml_metal_device_t dev);
void ggml_metal_moe_profile_free(ggml_metal_moe_profile_t profile);

void ggml_metal_moe_profile_reset(ggml_metal_moe_profile_t profile);
void ggml_metal_moe_profile_finish(ggml_metal_moe_profile_t profile);
struct ggml_metal_moe_profile_stats ggml_metal_moe_profile_get_stats(ggml_metal_moe_profile_t profile);
struct ggml_metal_moe_profile_bindings ggml_metal_moe_profile_get_bindings(
        ggml_metal_moe_profile_t profile,
        const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
