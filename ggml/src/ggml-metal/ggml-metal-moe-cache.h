#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_moe_cache * ggml_metal_moe_cache_t;
typedef struct ggml_metal_moe_profile * ggml_metal_moe_profile_t;

struct ggml_metal_moe_expert_source {
    int fd;
    uint64_t file_offset;
    uint64_t byte_size;
    int32_t type;
};

struct ggml_metal_moe_cache_result {
    bool hit;
    bool admitted;
    size_t offset;
};

struct ggml_metal_moe_cache_bindings {
    bool enabled;
    bool strict;
    bool strict_mm;
    struct ggml_metal_buffer_id weights;
    struct ggml_metal_buffer_id page_table;
    struct ggml_metal_buffer_id access;
};

ggml_metal_moe_cache_t ggml_metal_moe_cache_init(ggml_metal_device_t dev, size_t capacity_bytes);
void ggml_metal_moe_cache_free(ggml_metal_moe_cache_t cache);
bool ggml_metal_moe_cache_resize(ggml_metal_moe_cache_t cache, size_t capacity_bytes);
bool ggml_metal_moe_cache_set_source(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * tensor,
        struct ggml_metal_moe_expert_source source);
bool ggml_metal_moe_cache_needs_residency(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * tensor);
bool ggml_metal_moe_cache_same_bundle(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * first,
        const struct ggml_tensor * second);
bool ggml_metal_moe_cache_require(
        ggml_metal_moe_cache_t cache,
        const struct ggml_tensor * tensor,
        const int32_t * ids,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        size_t token_stride);

struct ggml_metal_moe_cache_result ggml_metal_moe_cache_access(ggml_metal_moe_cache_t cache, uint64_t key, size_t size_bytes);
struct ggml_metal_moe_cache_bindings ggml_metal_moe_cache_get_bindings(ggml_metal_moe_cache_t cache, const struct ggml_tensor * tensor);
void ggml_metal_moe_cache_update(ggml_metal_moe_cache_t cache);
void ggml_metal_moe_cache_attach_profile(ggml_metal_moe_cache_t cache, ggml_metal_moe_profile_t profile);
void ggml_metal_moe_cache_preload(ggml_metal_moe_cache_t cache, ggml_metal_moe_profile_t profile);
ggml_metal_buffer_t ggml_metal_moe_cache_get_buffer(ggml_metal_moe_cache_t cache);
size_t ggml_metal_moe_cache_get_capacity(ggml_metal_moe_cache_t cache);

#ifdef __cplusplus
}
#endif
