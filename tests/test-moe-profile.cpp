#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml.h"
#include "ggml-metal-device.h"
#include "ggml-metal-moe-profile.h"

#include <cassert>

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

    ggml_metal_moe_profile_t profile = ggml_metal_moe_profile_init(ggml_metal_device_get(0));
    assert(profile != nullptr);

    auto bindings = ggml_metal_moe_profile_get_bindings(profile, tensor);
    assert(bindings.enabled);
    assert(bindings.host_counts != nullptr);
    bindings.host_counts[1] = 3;
    bindings.host_counts[3] = 5;

    auto bindings2 = ggml_metal_moe_profile_get_bindings(profile, tensor2);
    assert(bindings2.enabled);
    assert(bindings2.counts.metal == bindings.counts.metal);
    assert(bindings2.counts.offs == bindings.counts.offs + 4 * sizeof(uint32_t));
    bindings2.host_counts[0] = 2;

    auto stats = ggml_metal_moe_profile_get_stats(profile);
    assert(stats.tensors == 2);
    assert(stats.selections == 10);
    ggml_metal_moe_tensor_profile_stats tensor_stats;
    assert(ggml_metal_moe_profile_get_tensor_stats(profile, 0, &tensor_stats));
    assert(tensor_stats.tensor == tensor);
    assert(tensor_stats.n_experts == 4);
    assert(tensor_stats.counts[1] == 3);
    assert(tensor_stats.counts[3] == 5);
    assert(!ggml_metal_moe_profile_get_tensor_stats(profile, 2, &tensor_stats));

    ggml_metal_moe_profile_reset(profile);
    stats = ggml_metal_moe_profile_get_stats(profile);
    assert(stats.tensors == 2);
    assert(stats.selections == 0);

    bindings.host_counts[2] = 7;
    ggml_metal_moe_profile_finish(profile);
    stats = ggml_metal_moe_profile_get_stats(profile);
    assert(stats.selections == 7);
    assert(!ggml_metal_moe_profile_get_bindings(profile, tensor).enabled);

    ggml_metal_moe_profile_free(profile);
    ggml_free(ctx);
    return 0;
}
