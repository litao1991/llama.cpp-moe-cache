# MoE cache milestone 1: prefill route profile

## Scope

Milestone 1 collects per-tensor, per-expert route counts during the initial prefill. It does not allocate or populate the expert cache during prefill. The profile is frozen immediately before the first decode step allocates the bounded expert cache.

## Design

- The Metal backend allocates one 1 MiB shared counter arena when an MoE cache is requested.
- Each expert weight tensor receives a slice with one 32-bit counter per expert.
- The matrix-matrix path adds the existing tokens-per-expert result to the counters.
- The matrix-vector path increments one counter per routed token and expert. Only one threadgroup performs the increment.
- A prefill starting at position zero resets old warmup data. Prefill chunks with increasing positions retain earlier counts.
- The first decode synchronizes Metal, freezes the profile, logs the total and the eight hottest experts for every tensor, and then allocates the expert cache.

The Qwen3.6-35B-A3B model uses 30,720 counters across 120 expert tensors. This is 120 KiB of the 1 MiB arena.

## Validation

### Automated tests

- `test-moe-cache`: passed.
- `test-moe-profile`: passed for arena allocation, tensor slicing, reset, totals, and freeze behavior.
- Metal `MUL_MAT_ID`: 792 of 792 supported numerical comparison cases passed. This covers matrix-vector and matrix-matrix dispatches.

### Qwen3.6-35B-A3B tests

Configuration:

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Input case: `case-01`, 2,048 effective prompt tokens
- Context: 4,096
- Ubatch: 512
- Cache budget: 256 MiB
- Metal: Apple M5

Observed profile:

| Metric | Result |
| --- | ---: |
| Expert tensors | 120 |
| MoE layers | 40 |
| Experts per layer | 256 |
| Routed experts per token | 8 |
| Selections per tensor | 16,384 |
| Total selections | 1,966,080 |
| Expected total | 1,966,080 |
| Active experts per layer, mean | 221.85 |
| Active experts per layer, range | 205 to 252 |
| Peak RSS | 2,883.6 MiB |
| Prompt throughput | 498.03 tokens/s |

The total is exact:

```text
2048 tokens * 40 layers * 8 routed experts * 3 expert tensors = 1,966,080
```

The gate, up, and down expert tensors in each layer reported identical route distributions, as required by the shared router decisions.

Longer decode tests used the same prompt, ubatch, and 256 MiB cache budget:

| Output tokens | Peak RSS | Prompt rate | Decode rate | Cache accesses | Cache hits | Hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3,469.0 MiB | 500.73 tokens/s | 20.80 tokens/s | 244,800 | 5,023 | 2.05% |
| 512 | 3,468.8 MiB | 505.15 tokens/s | 20.61 tokens/s | 490,560 | 8,460 | 1.72% |

The cache access totals are also exact. The first generated token uses the logits from prefill, so a request for `N` output tokens performs `N - 1` decode steps:

```text
(256 - 1) * 40 layers * 8 routed experts * 3 expert tensors = 244,800
(512 - 1) * 40 layers * 8 routed experts * 3 expert tensors = 490,560
```

Milestone 1 records prefill routes but does not use them for admission or preload. The low decode hit rates are therefore expected. Milestone 3 will preload hot experts from this profile.

## Metal allocation finding

The first implementation allocated one small Metal buffer per expert tensor. It passed small synthetic tests but caused an out-of-memory error on the full model because it also created 120 Metal residency sets. Replacing those buffers with one shared arena removed the error. The final implementation uses one Metal buffer and one residency set.

## Result

Milestone 1 passed its unit, backend numerical, real-model route-count, ubatch 512, and long-decode validation. No Metal out-of-memory error occurred with the final single-arena design.
