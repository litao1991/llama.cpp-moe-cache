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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>

static void test_strict_graph(uint32_t tokens, bool different_ids, ggml_type type = GGML_TYPE_F32) {
    const int width = type == GGML_TYPE_F32 ? 64 : 256;
    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);
    auto backend = ggml_backend_metal_init();
    assert(backend);
    auto reg = ggml_backend_metal_reg();
    using prepare_fn = bool (*)(ggml_backend_t, size_t);
    using source_fn = bool (*)(ggml_backend_t, const ggml_tensor *, int, uint64_t, uint64_t, int32_t);
    auto prepare = (prepare_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_prepare_moe_cache");
    auto set_source = (source_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_set_moe_source");
    assert(prepare && set_source);
    assert(prepare(backend, 192 * 1024));

    ggml_tensor * weights[3];
    ggml_tensor * outputs[3];
    const char * names[] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight" };
    auto input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 2, tokens);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, tokens);
    auto other_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, tokens);
    auto graph = ggml_new_graph(ctx);
    for (int part = 0; part < 3; ++part) {
        weights[part] = ggml_new_tensor_3d(ctx, type, width, 64, 4);
        ggml_set_name(weights[part], names[part]);
        outputs[part] = ggml_mul_mat_id(ctx, weights[part], input, part == 1 && different_ids ? other_ids : ids);
        ggml_build_forward_expand(graph, outputs[part]);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    char path[] = "/tmp/llama-moe-graph-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    std::vector<float> values(3 * 4 * 64 * width);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ((int) (i % 37) - 18) / 32.0f;
    }
    std::vector<char> packed(3 * ggml_nbytes(weights[0]));
    for (int part = 0; part < 3; ++part) {
        char * destination = packed.data() + part * ggml_nbytes(weights[part]);
        float * original = values.data() + part * 4 * 64 * width;
        if (type == GGML_TYPE_F32) {
            memcpy(destination, original, ggml_nbytes(weights[part]));
        } else {
            assert(ggml_quantize_chunk(type, original, destination, 0, 4 * 64, width, nullptr) == ggml_nbytes(weights[part]));
            ggml_get_type_traits(type)->to_float(destination, original, 4 * 64 * width);
        }
    }
    assert(pwrite(fd, packed.data(), packed.size(), 0) == (ssize_t) packed.size());
    std::vector<char> poison(ggml_nbytes(weights[0]), 0);
    for (int part = 0; part < 3; ++part) {
        ggml_backend_tensor_set(weights[part], poison.data(), 0, ggml_nbytes(weights[part]));
        assert(set_source(backend, weights[part], fd, part * ggml_nbytes(weights[part]), ggml_nbytes(weights[part]), type));
    }
    std::vector<float> inputs(width * 2 * tokens);
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = ((int) (i % 13) - 6) / 16.0f;
    }
    std::vector<int32_t> routes(2 * tokens), other_routes(2 * tokens);
    for (size_t i = 0; i < routes.size(); ++i) {
        routes[i] = i % 4;
        other_routes[i] = (i + 1) % 4;
    }
    ggml_backend_tensor_set(input, inputs.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(other_ids, other_routes.data(), 0, ggml_nbytes(other_ids));
    for (int repeat = 0; repeat < 2; ++repeat) {
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        for (int part = 0; part < 3; ++part) {
            std::vector<float> actual(64 * 2 * tokens);
            ggml_backend_tensor_get(outputs[part], actual.data(), 0, ggml_nbytes(outputs[part]));
            const auto & selected = part == 1 && different_ids ? other_routes : routes;
            double error = 0;
            double reference = 0;
            for (uint32_t token = 0; token < tokens; ++token) {
                for (int rank = 0; rank < 2; ++rank) {
                    const int expert = selected[token * 2 + rank];
                    for (int row = 0; row < 64; ++row) {
                        float expected = 0;
                        for (int column = 0; column < width; ++column) {
                            expected += values[part * 4 * 64 * width + expert * 64 * width + row * width + column] * inputs[(token * 2 + rank) * width + column];
                        }
                        const double difference = actual[(token * 2 + rank) * 64 + row] - expected;
                        assert(std::fabs(difference) < (type == GGML_TYPE_F32 ? 0.001 : 0.05));
                        error += difference * difference;
                        reference += expected * expected;
                    }
                }
            }
            assert(reference > 0 && error / reference < 5e-4);
        }
    }
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    close(fd);
    std::remove(path);
    ggml_free(ctx);
}

static void test_shared_file_cache(ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down) {
    char path[] = "/tmp/llama-moe-shared-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    ggml_tensor * members[] = { gate, up, down };
    constexpr size_t prefix = 32;
    std::vector<float> values(3 * ggml_nbytes(gate) / sizeof(float));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = (float) i + 0.5f;
    }
    const size_t bytes = values.size() * sizeof(float);
    assert(pwrite(fd, values.data(), bytes, prefix) == (ssize_t) bytes);

    auto cache = ggml_metal_moe_cache_init(ggml_metal_device_get(0), 16 * 1024);
    assert(cache != nullptr);
    assert(ggml_metal_buffer_is_shared(ggml_metal_moe_cache_get_buffer(cache)));
    for (size_t i = 0; i < 3; ++i) {
        const ggml_metal_moe_expert_source source = {
            fd, prefix + i * ggml_nbytes(gate), ggml_nbytes(members[i]), (int32_t) members[i]->type,
        };
        assert(ggml_metal_moe_cache_set_source(cache, members[i], source));
    }
    assert(ggml_metal_moe_cache_same_bundle(cache, gate, up));
    assert(ggml_metal_moe_cache_same_bundle(cache, gate, down));
    assert(!ggml_metal_moe_cache_same_bundle(cache, gate, nullptr));

    auto verify = [&](int32_t expert) {
        assert(ggml_metal_moe_cache_require(cache, gate, &expert, 1, 1, sizeof(expert)));
        const char * base = (const char *) ggml_metal_buffer_get_base(ggml_metal_moe_cache_get_buffer(cache));
        for (size_t i = 0; i < 3; ++i) {
            const char * expected = (const char *) values.data() + i * ggml_nbytes(gate) + expert * members[i]->nb[2];
            assert(memcmp(base + i * 4096, expected, members[i]->nb[2]) == 0);
        }
    };
    verify(0);
    verify(0);
    verify(1);
    verify(2);
    assert(ggml_metal_moe_cache_resize(cache, 32 * 1024));
    verify(2);
    assert(ggml_metal_moe_cache_resize(cache, 0));
    assert(ggml_metal_moe_cache_resize(cache, 16 * 1024));
    assert(ggml_metal_buffer_is_shared(ggml_metal_moe_cache_get_buffer(cache)));
    verify(0);

    assert(ftruncate(fd, prefix + bytes - 4) == 0);
    int32_t expert = 3;
    assert(!ggml_metal_moe_cache_require(cache, gate, &expert, 1, 1, sizeof(expert)));
    assert(pwrite(fd, values.data(), bytes, prefix) == (ssize_t) bytes);
    verify(3);
    ggml_metal_moe_cache_free(cache);
    close(fd);
    std::remove(path);
}

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
    test_shared_file_cache(gate, up, down);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    test_strict_graph(1, false);
    test_strict_graph(32, false);
    test_strict_graph(64, true);
    for (auto type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        test_strict_graph(1, false, type);
        test_strict_graph(32, false, type);
        test_strict_graph(64, true, type);
    }
    return 0;
}
