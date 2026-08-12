#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_moe_cache * ggml_metal_moe_cache_t;
typedef struct ggml_metal_moe_profile * ggml_metal_moe_profile_t;

struct ggml_metal_moe_cache_result {
    bool hit;
    bool admitted;
    size_t offset;
};

struct ggml_metal_moe_cache_bindings {
    bool enabled;
    struct ggml_metal_buffer_id weights;
    struct ggml_metal_buffer_id page_table;
    struct ggml_metal_buffer_id access;
};

ggml_metal_moe_cache_t ggml_metal_moe_cache_init(ggml_metal_device_t dev, size_t capacity_bytes);
void ggml_metal_moe_cache_free(ggml_metal_moe_cache_t cache);

struct ggml_metal_moe_cache_result ggml_metal_moe_cache_access(ggml_metal_moe_cache_t cache, uint64_t key, size_t size_bytes);
struct ggml_metal_moe_cache_bindings ggml_metal_moe_cache_get_bindings(ggml_metal_moe_cache_t cache, const struct ggml_tensor * tensor);
void ggml_metal_moe_cache_update(ggml_metal_moe_cache_t cache);
void ggml_metal_moe_cache_preload(ggml_metal_moe_cache_t cache, ggml_metal_moe_profile_t profile);
ggml_metal_buffer_t ggml_metal_moe_cache_get_buffer(ggml_metal_moe_cache_t cache);
size_t ggml_metal_moe_cache_get_capacity(ggml_metal_moe_cache_t cache);

#ifdef __cplusplus
}
#endif
