# P0 专家缓存改用 Shared Metal Buffer

日期：2026-11-11。基础分支：`codex/moe-out-of-core-p0`，HEAD `910fa24c5`。本文记录归档前的工作区验证，提交状态以 Git 历史为准。

## 方案

按[讨论中的 shared cache 方案](https://chatgpt.com/c/6af2fb25-a560-83e8-a694-7639e3a57e8a)，保留有限容量的专家缓存、SLRU、bundle、页表、prefill profile 和 strict residency，将缓存的存储类型从 private 改为 shared。

旧路径：

```text
GGUF -> pread -> 两个 32 MiB shared staging buffer
     -> Metal blit -> private expert cache -> GPU
```

新路径：

```text
GGUF -> pread 直接写 shared expert cache 的目标槽 -> GPU
```

CPU 和 GPU 访问同一份已加载专家数据。淘汰仍由缓存策略决定，系统不会因采用 shared 就随机覆盖专家槽位。

## 代码变化

- `ggml/src/ggml-metal/ggml-metal-moe-cache.mm`：初始化和 resize 均请求 shared buffer，并验证 CPU 可访问；不再分配 private 专家缓存、双 staging 或 cache blit command buffer。
- 文件源通过 `pread` 直接写 cache base + slot offset；处理短读和 EINTR。非文件的 shared tensor 来源使用 CPU memcpy 直接写缓存。
- 完整 bundle 读取成功后才发布页表。读取失败时撤销当前槽与页表，使 strict require 返回失败；预取/预装失败也不会留下有效的半成品映射。
- 没有异步 blit backlog 后，adaptive controller 的 copy-backlog 输入为 false；其他需求淘汰、useful/wasted、剩余 token 控制保留。
- 保留旧 transfer 统计字段供历史解析器识别，`staging batches`、`async copies`、`direct bytes`、`staged bytes` 均为 0；新增 `file read bytes`、`host copy bytes`、`read failures`，不能再把 staged bytes 当成新版本的文件读取量。
- `ggml-metal-moe-cache.h` 记录 shared 槽位修改前必须结束 GPU 读取的调用约束。

本次没有移入性能分支的“每层一次 residency”、strict MM、下一层后台 worker 等优化；仍保留 P0 的逐专家矩阵图边界，便于独立评估存储路径变化。

## CPU/GPU 同步

P0 strict 图在每段计算结束后执行 `waitUntilCompleted`。CPU 只有在此后才能淘汰旧槽、写入新权重、更新页表，然后提交下一段计算。当前 `pread` 是同步的，提交前已写完目标专家。

非 strict 的 cache update 在 `ggml_metal_synchronize` 等待 GPU 后执行；resize 同样由 backend 先 synchronize。mutex 只保护 CPU 元数据，不能替代这些 GPU 完成等待。

因此可以删除 cache 的复制队列等待，但不能连同图执行中的 GPU 等待一起删除。未来接入异步预取时，需要重新管理 in-flight 槽位和写入/计算依赖。

## 验证时发现的模型加载问题

历史模型路径已经不存在，本机可用文件是：

```text
/Users/litao/Documents/工具/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

此 GGUF 含有可选 MTP 层。普通推理没有创建该层 tensor，但 P0 遍历全部专家元数据注册文件来源，因 `blk.40.ffn_down_exps.weight` 不存在于已加载模型而失败。

`src/llama-model.cpp` 现在只为已经创建的专家 tensor 注册来源，跳过 loader 有意省略的可选 tensor，fd 复制失败仍报错。

private 和 shared 对照都使用相同的 loader 修复。private 测试通过保存的原版 Metal dylib 执行，shared 使用修改后的 Metal dylib，避免把 loader 兼容修复当成缓存性能收益。

## 构建与测试

Release Metal 构建 `llama-completion`、`test-moe-cache`、`test-moe-profile` 通过。policy 测试、Metal profile/shared-cache 集成测试以及 diff 格式检查通过。

新增集成测试覆盖：

1. shared buffer 的 CPU 可访问性。
2. 非零文件偏移、gate/up/down 三个来源和目标 bundle 的数据逐字节比对。
3. 重复命中与容量不足时的专家替换。
4. resize、释放到 0、重新创建后的读取正确性。
5. 文件尾截断导致短读/EOF，require 返回失败；修复文件后重新读取成功。

测试日志里的 `failed to read expert 3` 属于第 5 项主动注入的错误，不是测试失败。真实 Qwen 运行的 read failures 均为 0。

构建命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_OPENSSL=OFF -DGGML_CCACHE=OFF
cmake --build build --target llama-completion test-moe-cache test-moe-profile -j 8
build/bin/test-moe-cache
build/bin/test-moe-profile
```

Metal 测试需要有正常 Metal 设备访问权限；本次沙箱内分配失败，沙箱外运行通过。

## 真实 Qwen 对照

平台：Apple M5、macOS 26.6.2、Metal。模型为上述含 MTP 权重的 Q4_K_M 文件，普通推理不启用 MTP。

公共参数：缓存 1024 MiB，context 2048，batch/ubatch 64，threads/batch threads 4，prompt 为 `Explain why the sky is blue in one sentence.`，生成 128 token，忽略 EOS，temp 0，seed 42，禁用 warmup。

顺序为 `private -> shared -> shared -> private`。成功运行日志编号依次是 `private-02`、`shared-01`、`shared-02`、`private-03`。`private-01` 是 loader 修复前的失败诊断，不纳入性能数据。

| 指标 | Private | Shared |
|---|---:|---:|
| 两轮 decode | 5.43 / 5.48 tok/s | 6.00 / 6.22 tok/s |
| 平均 decode | 5.455 tok/s | 6.110 tok/s |
| 两轮 prompt throughput | 3.13 / 6.96 tok/s | 6.40 / 7.15 tok/s |
| 峰值 footprint | 1356.72 / 1343.57 MiB | 1274.32 / 1280.16 MiB |
| 平均峰值 footprint | 1350.15 MiB | 1277.24 MiB |
| 峰值 RSS | 2763.86 / 2795.50 MiB | 3748.16 / 3745.63 MiB |
| 平均 system CPU time | 6.865 s | 6.590 s |
| 每轮同步专家加载 | 25,727 | 25,727 |
| 每轮 demand accesses / hits | 40,640 / 34,834 | 40,640 / 34,834 |
| 每轮 staging / file read bytes | 56,083,349,504 | 56,083,349,504 |
| 每轮 cache blit operations | 88,093 | 0 |
| 每轮 cache copy batches | 5,415 | 0 |
| 物理分配失败 | 0 | 0 |

四份生成输出完全一致。各轮策略准入、淘汰、加载和命中计数相同，累计文件区间读取量相同，说明本次差异主要来自内存访问和复制路径。

decode 均值约提升 **12.0%**；与最后一轮热态 private 的 5.48 tok/s 相比，shared 两轮分别高约 9.5% 和 13.5%。但只有一个短 prompt、每种两轮，不能据此承诺所有预算和长输入都提升 12%。第一轮 private prefill 明显较冷，不适合将其 prefill 差值当作稳定收益。

平均峰值 footprint 减少约 **72.91 MiB**，符合移除固定 64 MiB staging 及其管理开销的方向。RSS 则增加约 0.95 GiB，因为最终缓存现在由 CPU 直接写入，进程驻留记账发生变化；不能将 RSS 增长直接当成同等的总物理内存增长，也不能宣称此修改减少了 1 GiB 缓存本体。

每轮约 52.23 GiB 的 `file read bytes` 是累计 pread 字节，可能命中操作系统文件缓存，不是 SSD 物理读取量。此次消除的是 staging -> private 的显式 cache blit，未消除文件读取、专家缓存容量或 P0 图同步成本。

## 复现与结果位置

当前 shared 版本可执行：

```bash
build/bin/llama-completion \
  -m '/Users/litao/Documents/工具/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf' \
  --moe-cache-size 1024 -c 2048 -b 64 -ub 64 -t 4 -tb 4 \
  -p 'Explain why the sky is blue in one sentence.' -n 128 \
  --ignore-eos --seed 42 --temp 0 --no-display-prompt --no-conversation --no-warmup
```

逐轮日志、输出文本和 `/usr/bin/time -l` 位于仓库下 `moe-cache-eval/shared-cache-20261111/`。private 对照程序及 dylib 临时保存在 `/tmp/moe-shared-ab.Ou9wMq/private-bin/`，该目录不是长期归档；其 Metal cache 对应 HEAD `910fa24c5`，loader 使用本次可选 tensor 修复。

`GGML_METAL_MOE_CACHE_DIRECT_BLIT` 在新 shared cache 中不再控制复制路径。页表和有限容量缓存仍保留；不支持 CPU 可写 shared buffer 的设备会拒绝创建本缓存，而不会悄悄回退 private。

后续需要补充多用例、256/512-token、256 MiB/512 MiB 等预算对照，以及 iPhone 真机的 footprint、内存警告和前后台测试。本轮没有进行 iPhone 验证，也没有 commit 或 push。
