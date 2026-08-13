#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"
#include "ggml-metal-moe-cache.h"
#include "ggml-metal-moe-profile.h"

#include <cassert>
#include <cstdio>
#include <unistd.h>

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    ggml_set_name(tensor, "test.experts");
    ggml_tensor * tensor2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 3);
    ggml_set_name(tensor2, "test.experts.2");
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");

    ggml_metal_moe_profile_t profile = ggml_metal_moe_profile_init(ggml_metal_device_get(0));
    assert(profile != nullptr);

    auto bindings = ggml_metal_moe_profile_get_bindings(profile, tensor);
    assert(bindings.enabled);
    assert(bindings.transitions_enabled);
    assert(bindings.host_counts != nullptr);
    assert(bindings.host_transitions != nullptr);
    assert(bindings.host_previous != nullptr);
    bindings.host_counts[1] = 3;
    bindings.host_counts[3] = 5;
    assert(bindings.transition_slots > 0);
    bindings.host_transitions[1 * bindings.transition_slots] = (7u << 16) | (3u + 1);
    bindings.host_previous[0] = 2;

    auto bindings2 = ggml_metal_moe_profile_get_bindings(profile, tensor2);
    assert(bindings2.enabled);
    assert(bindings2.counts.metal == bindings.counts.metal);
    assert(bindings2.counts.offs == bindings.counts.offs + 4 * sizeof(uint32_t));
    bindings2.host_counts[0] = 2;

    auto gate_bindings = ggml_metal_moe_profile_get_bindings(profile, gate);
    auto up_bindings = ggml_metal_moe_profile_get_bindings(profile, up);
    auto down_bindings = ggml_metal_moe_profile_get_bindings(profile, down);
    assert(gate_bindings.enabled);
    assert(!up_bindings.enabled);
    assert(!down_bindings.enabled);
    assert(gate_bindings.host_counts == up_bindings.host_counts);
    assert(gate_bindings.host_counts == down_bindings.host_counts);
    gate_bindings.host_counts[2] = 11;

    auto stats = ggml_metal_moe_profile_get_stats(profile);
    assert(stats.tensors == 5);
    assert(stats.selections == 21);
    ggml_metal_moe_tensor_profile_stats tensor_stats;
    assert(ggml_metal_moe_profile_get_tensor_stats(profile, 0, &tensor_stats));
    assert(tensor_stats.tensor == tensor);
    assert(tensor_stats.n_experts == 4);
    assert(tensor_stats.counts[1] == 3);
    assert(tensor_stats.counts[3] == 5);
    assert(tensor_stats.transition_slots == bindings.transition_slots);
    assert(tensor_stats.transitions[1 * tensor_stats.transition_slots] == ((7u << 16) | (3u + 1)));
    assert(ggml_metal_moe_profile_get_tensor_stats(profile, 3, &tensor_stats));
    assert(tensor_stats.tensor == up);
    assert(tensor_stats.counts[2] == 11);
    assert(!ggml_metal_moe_profile_get_tensor_stats(profile, 5, &tensor_stats));

    ggml_metal_moe_profile_reset(profile);
    stats = ggml_metal_moe_profile_get_stats(profile);
    assert(stats.tensors == 5);
    assert(stats.selections == 0);
    assert(bindings.host_transitions[1 * bindings.transition_slots] == 0);
    assert(bindings.host_previous[0] == UINT32_MAX);

    bindings.host_counts[2] = 7;
    ggml_metal_moe_profile_finish(profile);
    stats = ggml_metal_moe_profile_get_stats(profile);
    assert(stats.selections == 7);
    assert(!ggml_metal_moe_profile_get_bindings(profile, tensor).enabled);

    ggml_metal_moe_profile_release_transitions(profile);
    assert(ggml_metal_moe_profile_get_tensor_stats(profile, 0, &tensor_stats));
    assert(tensor_stats.transitions == nullptr);

    ggml_metal_moe_profile_free(profile);

    ggml_backend_t backend = ggml_backend_metal_init();
    assert(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    ggml_metal_moe_cache_t cache = ggml_metal_moe_cache_init(ggml_metal_device_get(0), 1024 * 1024);
    assert(cache != nullptr);
    const auto gate_cache = ggml_metal_moe_cache_get_bindings(cache, gate);
    const auto up_cache = ggml_metal_moe_cache_get_bindings(cache, up);
    const auto down_cache = ggml_metal_moe_cache_get_bindings(cache, down);
    assert(gate_cache.enabled);
    assert(up_cache.enabled);
    assert(down_cache.enabled);
    assert(gate_cache.access.metal == up_cache.access.metal);
    assert(gate_cache.access.offs == up_cache.access.offs);
    assert(gate_cache.access.metal == down_cache.access.metal);
    assert(gate_cache.access.offs == down_cache.access.offs);
    assert(gate_cache.page_table.metal != up_cache.page_table.metal);
    assert(gate_cache.page_table.metal != down_cache.page_table.metal);
    assert(ggml_metal_moe_cache_resize(cache, 512 * 1024));
    assert(ggml_metal_moe_cache_get_capacity(cache) == 512 * 1024);
    assert(ggml_metal_moe_cache_get_bindings(cache, gate).enabled);
    assert(ggml_metal_moe_cache_resize(cache, 0));
    assert(ggml_metal_moe_cache_get_capacity(cache) == 0);
    assert(!ggml_metal_moe_cache_get_bindings(cache, gate).enabled);
    assert(ggml_metal_moe_cache_resize(cache, 256 * 1024));
    assert(ggml_metal_moe_cache_get_bindings(cache, gate).enabled);

    char path[] = "/tmp/llama-moe-source-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(ftruncate(fd, (off_t) ggml_nbytes(gate)) == 0);
    ggml_metal_moe_expert_source source = {
        fd,
        0,
        ggml_nbytes(gate),
        (int32_t) gate->type,
    };
    assert(ggml_metal_moe_cache_set_source(cache, gate, source));
    close(fd);
    std::remove(path);

    ggml_metal_moe_cache_free(cache);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
