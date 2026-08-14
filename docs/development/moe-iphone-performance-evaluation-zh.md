# MoE iPhone 基线性能优化评估

## 结论

本轮按既定顺序完成了五项实现和真实 Qwen A/B 验证。当前建议配置是 1 GiB 专家缓存、每层一次 residency、strict `mul_mm_id`、`ubatch=64`，下一层异步预取默认关闭。

最有价值的两个代码优化是：

1. 每层一次 residency 将 1 GiB decode 从 8.25 提升到 11.74 tok/s，提升 42.3%，没有增加 footprint 或文件读取量。
2. strict `mul_mm_id` 将 101-token prefill 从 41.22 提升到 59.35 tok/s，提升 44.0%，decode 和内存基本不变。

把缓存从 1 GiB 继续增加到 1.25 GiB，虽然路由缓存命中率由 87.78% 提高到 94.00%，文件读取量由 28.05 GiB 降到 24.41 GiB，但配对样本的 decode 由 11.87 降到 11.04 tok/s，同时 footprint 增加 257.30 MiB。因此 1.25 GiB 当前没有足够的速度边际收益。

## 测试条件

- 模型：`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- 平台：macOS，Apple M5，Metal
- context：2048
- batch / ubatch：64 / 64；256 MiB 基线因容量约束自动降到 ubatch 16
- decode 用例：11-token prompt，固定生成 64 token，`seed=42`，`temp=0`，忽略 EOS
- prefill 用例：101-token prompt，固定生成 8 token
- 主性能指标：进程内 `common_perf_print`
- 内存指标：`/usr/bin/time -l` 的 maximum RSS 和 peak memory footprint
- 读取量：缓存统计中的 `staged bytes`，代表从专家文件进入 staging 的累计字节数

每个配置运行一个固定样本。结果适合判断大幅优化和明确回退；小于约 3% 的差异仍应通过多轮、交错顺序和温度稳定后的测试确认。

## 1. 缓存从 256 MiB 增加到 1 GiB

| 指标 | 256 MiB | 1 GiB | 变化 |
|---|---:|---:|---:|
| prompt | 16.98 tok/s | 22.13 tok/s | +30.3% |
| decode | 5.56 tok/s | 8.25 tok/s | +48.4% |
| 路由缓存命中率 | 4.26% | 87.78% | +83.52 pp |
| 同步专家加载 | 21,637 | 14,244 | -34.2% |
| 文件读取量 | 62.38 GiB | 28.05 GiB | -55.0% |
| peak footprint | 596.30 MiB | 1,333.96 MiB | +737.66 MiB |
| peak RSS | 2,779.00 MiB | 2,774.70 MiB | -4.30 MiB |

收益来自更高的专家复用和更少的文件读取。代价主要体现在 `phys_footprint`，而不是 RSS；这再次说明 iPhone 容量决策不能只看 RSS。

## 2. 每层一次 residency 和全命中快速路径

原实现把每个 strict `MUL_MAT_ID` 都作为图边界。Qwen 每层包含 gate、up、down 三个同 bundle 专家张量，因此每个 token 有约 120 次提交和等待。

`ggml_metal_graph_compute()` 现在通过 `ggml_metal_moe_cache_same_bundle()` 识别同一 bundle，把三个专家矩阵及其间节点放在同一个命令缓冲中。每个 token 只在 40 个层边界读取路由并确保 residency。`ggml_metal_moe_cache_require()` 还会区分需要复制的调用和整次全命中的调用；全命中且没有未完成预取时，不再等待复制队列。

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| prompt | 22.13 tok/s | 24.10 tok/s | +8.9% |
| decode | 8.25 tok/s | 11.74 tok/s | +42.3% |
| 总推理时间 | 8.15 s | 5.84 s | -28.4% |
| residency 调用 | 约 7,680 | 2,560 | -66.7% |
| 全命中 residency | 未统计 | 25 | 新增 |
| 文件读取量 | 28.05 GiB | 28.05 GiB | 不变 |
| peak footprint | 1,333.96 MiB | 1,333.38 MiB | 基本不变 |

该方法只减少同步和命令提交，不改变专家选择、缓存准入或文件流量，因而是当前风险最低、decode 收益最大的优化。

## 3. 恢复 strict `mul_mm_id` 和 ubatch 64

strict 模式原先强制走 `mul_mv_id`。本轮给 `kernel_mul_mm_id` 增加了 strict 页表参数：内核按专家 ID 从 page table 取得 cache offset，遇到无效页表项立即返回，并在 access buffer 标记实际使用专家。CPU 端在 `ggml_metal_op_mul_mat_id()` 中重新允许 strict 模式在 token 数不少于 32 时选择矩阵矩阵内核。

环境变量 `GGML_METAL_MOE_CACHE_MM_ID=0` 可以回退到 strict `mul_mv_id`，用于同一二进制 A/B。

| 指标 | strict mul_mv_id | strict mul_mm_id | 变化 |
|---|---:|---:|---:|
| 101-token prefill | 41.22 tok/s | 59.35 tok/s | +44.0% |
| decode | 12.15 tok/s | 12.46 tok/s | +2.6% |
| 文件读取量 | 14.20 GiB | 14.18 GiB | -0.1% |
| peak footprint | 1,339.49 MiB | 1,335.51 MiB | 基本不变 |
| peak RSS | 2,780.84 MiB | 2,780.97 MiB | 基本不变 |

代价是 Metal ABI 多一个 strict 标记、page table 和 access buffer 绑定，以及每个专家 threadgroup 的页表读取。短 decode 不满足矩阵矩阵阈值，因此没有显著收益或回退。

## 4. 下一层异步预取

实现包含三部分：

- `ggml_moe_markov::observe_transition()` 学习本层专家集合到下一层专家集合的统计关系。
- 专家复制使用独立 Metal command queue，使下一层专家 I/O 可以与当前层计算重叠。
- 带 protected key 的 prefetch 准入，保证预取不会淘汰当前层即将执行的专家。

可使用以下环境变量实验：

- `GGML_METAL_MOE_CACHE_NEXT_LAYER_PREFETCH=1`
- `GGML_METAL_MOE_CACHE_NEXT_LAYER_MIN_PERCENT=<percent>`
- `GGML_METAL_MOE_CACHE_NEXT_LAYER_MIB=<MiB>`

4% 门槛、4 MiB 上限的 A/B 结果：

| 指标 | 关闭 | 开启 | 变化 |
|---|---:|---:|---:|
| decode | 11.57 tok/s | 11.39 tok/s | -1.6% |
| 同步专家加载 | 14,244 | 13,109 | -8.0% |
| 文件读取量 | 28.05 GiB | 30.71 GiB | +9.5% |
| 下一层预取准入 | 0 | 2,504 | +2,504 |
| 有用 / 浪费预取 | 1 / 7 | 963 / 1,568 | 有效率约 38% |
| peak footprint | 1,335.11 MiB | 1,335.24 MiB | 基本不变 |

13% 和 15% 门槛下预取次数均为 0。原因是当前统计使用每层专家集合，Qwen top-8 路由形成的跨层概率集中在约 12.5%；低门槛会同时选择大量候选，高门槛则全部过滤。该方案已实现但默认关闭，避免把负收益带到基线。

后续若要重新启用，应从 GPU 保留 token/rank 级路由序列，并学习按 token 对齐的稀疏跨层转移，而不是只学习集合共现。

## 5. 1 GiB 与 1.25 GiB 最终比较

| 指标 | 1 GiB | 1.25 GiB | 变化 |
|---|---:|---:|---:|
| prompt | 24.48 tok/s | 24.56 tok/s | +0.3% |
| decode | 11.87 tok/s | 11.04 tok/s | -7.0% |
| 路由缓存命中率 | 87.78% | 94.00% | +6.22 pp |
| 同步专家加载 | 14,244 | 13,035 | -8.5% |
| 全命中 residency | 25 | 44 | +76.0% |
| 文件读取量 | 28.05 GiB | 24.41 GiB | -13.0% |
| peak footprint | 1,335.08 MiB | 1,592.38 MiB | +257.30 MiB |
| peak RSS | 2,785.20 MiB | 2,786.50 MiB | +1.30 MiB |

更大缓存改善了所有缓存和 I/O 指标，但没有转化为本次 decode 速度收益。可能原因包括更大的 Metal working set、运行噪声和同步加载已不再是唯一瓶颈。对 iPhone 而言，额外 257 MiB footprint 的代价是确定的，而速度收益并不成立，所以推荐停在 1 GiB；更保守的设备应从 512 MiB 或更低起步并动态扩容。

## 代码与验证范围

主要修改位置：

- `ggml/src/ggml-metal/ggml-metal-context.m`：按 bundle 合并 strict 图段。
- `ggml/src/ggml-metal/ggml-metal-moe-cache.mm`：residency 快速路径、独立复制队列、跨层预取与统计。
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`、`ggml-metal.metal`、`ggml-metal-impl.h`：strict `mul_mm_id` 页表支持。
- `ggml/src/ggml-moe-cache.cpp`：受保护的 prefetch 准入和跨层 Markov 观察。
- `tests/test-moe-cache.cpp`：受保护淘汰与跨层预测单元测试。

验证已完成：

- Release 构建 `llama-completion` 和 `test-moe-cache`。
- `test-moe-cache` 全部通过。
- `git diff --check` 通过。
- 真实 Qwen Metal prefill、decode、1 GiB/1.25 GiB 和预取开关 A/B 均正常退出，无 Metal command buffer 错误，无 swap。

本轮未提交或 push；所有改动位于本地分支 `codex/moe-iphone-performance`。
