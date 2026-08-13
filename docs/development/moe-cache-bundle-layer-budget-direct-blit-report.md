# MoE cache bundle, layer budget, and direct blit report

## Summary

This change implements three follow-ups selected after reviewing [Patterns behind Chaos](https://arxiv.org/abs/2510.05497):

1. Cache the gate, up, and down weights of one logical expert as one bundle.
2. Allocate layer watermarks from prefill demand and bundle read cost instead of equal shares.
3. Blit mmap-backed shared Metal buffers directly into the private cache without a CPU staging copy.

The implementation keeps the existing `--moe-cache-size` interface. Each feature has a same-binary environment switch for validation and rollback.

## Design

### Logical expert bundles

The cache recognizes `blk.N.ffn_gate_exps`, `blk.N.ffn_up_exps`, and `blk.N.ffn_down_exps` tensors with the same layer and expert count. It creates one bundle per `(layer, expert)` and keeps these items at bundle scope:

- one access bitmap;
- one Markov predictor;
- one SLRU and slot-allocation key;
- one admission and eviction decision.

Each tensor keeps its own page table because the Metal kernel still receives one weight tensor at a time. A successful bundle admission copies all member tensors into one aligned private-cache range and publishes every page-table entry together. Eviction invalidates all member page tables together.

The prefill profiler uses the same logical grouping. Only the first tensor in a gate/up/down group records route counts and transitions. The other tensors expose the shared host statistics but do not run duplicate atomic updates in the Metal kernel.

Unknown tensor names and duplicate parts remain standalone bundles. Setting `GGML_METAL_MOE_CACHE_BUNDLE=0` restores per-tensor profiles and cache entries.

### Dynamic layer budgets

The prefill candidates are converted to layer-budget items with this score:

```text
score = expected prefill hits * uncached bundle read bytes / aligned bundle bytes
```

The allocator first reserves the hottest fitting bundle for every active layer. It then assigns each additional bundle to the highest remaining marginal score until capacity is exhausted. Residual bytes are added to the layer with the largest assigned budget. The result is installed as the existing per-group soft watermark, so global capacity can still be used when another layer does not consume its share.

Setting `GGML_METAL_MOE_CACHE_DYNAMIC_LAYER_BUDGET=0` restores equal layer watermarks.

### Direct shared-buffer blit

The model weights already reside in mmap-backed shared `MTLBuffer` objects. The direct path resolves the source buffer and tensor offset once during tensor registration, then encodes this copy for every admitted expert part:

```text
shared model MTLBuffer -> private MoE cache MTLBuffer
```

This removes the CPU `memcpy` and the two fixed 32 MiB staging buffers. Copy command buffers remain double-buffered so cache updates do not overwrite a command buffer that is still in flight.

Setting `GGML_METAL_MOE_CACHE_DIRECT_BLIT=0` allocates the old staging buffers and restores the previous mmap-to-CPU-to-private path.

## Same-binary switches

| Variable | Default | Effect when set to `0` |
| --- | --- | --- |
| `GGML_METAL_MOE_CACHE_BUNDLE` | `1` | Use separate profile, policy, and cache state for every tensor |
| `GGML_METAL_MOE_CACHE_DYNAMIC_LAYER_BUDGET` | `1` | Divide capacity equally across layers |
| `GGML_METAL_MOE_CACHE_DIRECT_BLIT` | `1` | Copy through two 32 MiB CPU staging buffers |

The earlier prefetch, adaptive controller, prefill Markov, and remaining-token switches are unchanged.

## Validation

### Build and unit tests

These targets build successfully on Apple Metal:

```text
test-moe-cache
test-moe-profile
llama-cli
```

`test-moe-cache` now covers custom group budgets and layer water-filling in addition to the existing SLRU, slot allocator, admission cost, Markov, and prefetch controller tests.

`test-moe-profile` verifies that gate/up/down tensors share one profile and one cache access bitmap while retaining distinct page tables. The Metal test must run outside the filesystem sandbox because the sandbox cannot allocate an Apple Metal buffer.

### Real-model smoke test

A 2,048-token prompt and 16-token decode completed with a 512 MiB cache on Apple M5 using Qwen3.6-35B-A3B-UD-Q4_K_M. The run created 40 bundles from 120 expert tensors, preloaded 280 logical experts, seeded 40 Markov matrices, reported 85 cost rejections and zero allocation failures, and copied 3,679,846,400 bytes through direct blits with zero staged bytes.

### Same-binary ablation

The focused benchmark used the same model and case-01 prompt, 64 output tokens, a 1,024 MiB cache, temperature 0, and seed 20260812. Every comparison below is the median of three interleaved runs. All 24 real-model runs across the full set of diagnostics completed successfully.

#### Bundle isolation with direct blit

| Bundle | Logical profiles | Cache accesses | Hit rate | Predictions | Decode tok/s | Copied bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 120 | 60,480 | 31.49% | 60,480 | 30.35 | 9,054,027,776 |
| on | 40 | 20,160 | 35.17% | 20,160 | 29.82 | 13,331,169,280 |

Bundling reduced profile updates, cache policy accesses, and predictions by exactly 3x. The logical hit rate increased by 3.68 percentage points. In this short run, decode throughput decreased by 1.75% because a bundle miss copied all three parts and increased copied bytes by 47.24%. The tradeoff is deliberate: partial logical experts can no longer be resident.

#### Dynamic budget isolation with bundle and direct blit

| Layer budget | Budget range | Preloaded bundles | Hit rate | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| equal | 25.60-25.60 MiB | 557 | 35.17% | 18.83 |
| dynamic | 12.69-36.49 MiB | 561 | 34.72% | 18.95 |

The dynamic allocator produced nonuniform watermarks and preserved a minimum bundle for every layer. In this single prompt, it changed decode throughput by +0.64% but reduced hit rate by 0.45 percentage points. Prefill frequency is therefore not yet a reliable decode budget signal by itself. This mechanism needs a repeated multi-prompt matrix before its default policy is considered tuned.

#### Direct blit isolation with bundle and dynamic budget

| Copy path | Copy batches | Copy operations | Staged bytes | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| CPU staging | 441 | 21,496 | 13,516,955,648 | 21.77 |
| direct blit | 64 | 21,225 | 0 | 26.42 |

Direct blit removed all CPU staging bytes, removed the fixed 64 MiB staging allocation, reduced copy batches by 85.49%, and increased median decode throughput by 21.36% in the interleaved diagnostic. This is the strongest measured result of the three changes.

## Limits and follow-ups

- The performance data is a focused diagnostic from one prompt, not the previous 10-input by 6-budget matrix.
- Bundle misses can copy more total data even when the bundle hit rate is higher. A later policy can admit a bundle only when the predicted reuse offsets its full copy cost.
- Dynamic budgets currently use prefill frequency and read bytes. Adding per-layer decode usefulness feedback or transition entropy should make the allocation less prompt-specific.
- The direct path requires each expert tensor to resolve to one shared Metal buffer range. Registration disables the cache for a tensor when this invariant is not satisfied.
- The environment switches are backend-local experimental controls, not stable public configuration.
