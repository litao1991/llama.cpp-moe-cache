# MoE cache prefill transition and adaptive prefetch report

## Summary

This change implements the two P0 follow-ups selected after reviewing [Patterns behind Chaos](https://arxiv.org/abs/2510.05497):

1. Seed the decode Markov predictor from cross-token expert transitions observed during prefill.
2. Add adaptive prefetch throttling and price the complete capacity eviction set with one benefit unit.

The implementation keeps the existing `--moe-cache-size` interface. All new behavior is internal to the experimental Metal MoE cache and can be disabled with environment variables for same-binary ablation.

## Design

### Prefill transition seed

The Metal `MUL_MAT_ID` map kernel now records a dense `uint32_t` transition matrix for each expert tensor. Each token thread records its top-k to top-k transition with atomic increments. The profile keeps the final top-k expert set from each prefill chunk and includes the transition across chunk boundaries.

The profile has a 64 MiB shared transition arena. A tensor uses `n_experts * n_experts + n_experts` entries for its matrix and previous state. If the arena cannot fit another tensor, that tensor retains the old frequency-only profile behavior. The arena is released after cache preload and Markov seeding.

At the first decode boundary, every available transition row is normalized to a prior strength of 32. Counts that round to zero are omitted, which prevents the long low-probability tail of a large prefill from overpowering online decode observations. The seed lets the first routed decode token produce predictions immediately. Online observations then update the same matrix.

### Full eviction-set cost

Admission now simulates the complete victim set needed to make capacity for a candidate. The simulation follows the same soft group budget and SLRU victim order as real eviction and updates simulated group residency after each selected victim.

Candidate and victim values use the same fixed-point unit:

```text
value = expected accesses * 1024 * expert bytes
```

Demand candidates use `(4 * short_frequency + long_frequency) / 5`, so one new demand access is one normalized access. Predicted candidates use transition probability directly. Both are scaled by 1024 before admission requires the candidate value to exceed the sum of victim values by the existing 1.25 safety factor.

The preload path now uses the prefetch API instead of two synthetic demand accesses. This avoids assigning artificial demand benefit to every preloaded expert.

### Adaptive prefetch

The controller adjusts the byte budget and minimum probability from four signals:

- rolling useful and wasted prefetch outcomes;
- demand eviction pressure in the current decode update;
- staging buffer copy pressure;
- an optional remaining-token hint.

The controller starts with a small exploration budget. After at least eight outcomes, it uses these usefulness bands:

| Useful ratio | Byte budget | Minimum probability |
| ---: | ---: | ---: |
| at least 50% | 100% | 4% |
| 25% to 50% | 50% | 8% |
| 12.5% to 25% | 25% | 12% |
| below 12.5% | one probe every 16 routed updates | 20% |

Demand eviction and copy pressure reduce the byte budget further. Four or fewer remaining tokens raise the threshold to 20%; one or fewer disables prefetch. The controller advances only when a synchronization observed real MoE routes. Empty backend synchronizations do not consume the remaining-token hint.

## Same-binary switches

| Variable | Default | Effect when set to `0` |
| --- | --- | --- |
| `GGML_METAL_MOE_CACHE_PREFETCH` | `1` | Disable predicted prefetch |
| `GGML_METAL_MOE_CACHE_ADAPTIVE_PREFETCH` | `1` | Use the fixed 8 MiB or capacity/64 budget and 4% threshold |
| `GGML_METAL_MOE_CACHE_PREFILL_MARKOV` | `1` | Do not collect or seed Markov matrices from prefill |

`GGML_METAL_MOE_CACHE_REMAINING_TOKENS=N` supplies an optional request-length hint for one-shot runs. Without it, the controller still uses feedback, eviction pressure, and copy pressure. A backend-local environment variable was used because the Metal backend does not receive a request-level `n_predict` value. Long-running servers must not set this hint because the backend has no request-boundary reset signal yet.

## Validation

### Build and unit tests

The following targets build successfully:

```text
test-moe-cache
test-moe-profile
llama-cli
```

`test-moe-cache` covers:

- a candidate that needs a cold first victim and a hot second victim, proving that the full set is priced;
- cost rejection accounting;
- adaptive warmup, copy pressure, eviction pressure, and remaining-token limits;
- prefill Markov seed ranking before any decode transition is observed.

The 64-token ablation and a separate 256-token, 512 MiB fixed-prefetch stress run reported zero real-model cost rejections. In these runs, every required victim set still contained enough preloaded or prefetched entries with zero demand history to make its aggregate loss zero. The focused unit test is the validation that the nonzero multi-victim rejection branch works; a zero production counter is not treated as proof by itself.

`test-moe-profile` runs on Apple Metal and covers transition buffer allocation, host access, reset, stats exposure, and release.

### Real model smoke

The final binary completed a 2,039-token prompt and 16 output tokens with a 512 MiB cache on Apple M5. It seeded all 120 Qwen expert tensors, completed with no Metal error, and reported no allocation failure.

### Same-binary ablation

The diagnostic used Qwen3.6-35B-A3B-UD-Q4_K_M, case-01 with 2,039 source tokens, 64 output tokens, a 1,024 MiB cache, temperature 0, and the same binary for every row.

| Prefill seed | Adaptive | Hit rate | Decode tok/s | Predictions | Prefetch attempts | Useful | Wasted | Useful ratio | Async copies |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | off | 30.98% | 25.10 | 58,920 | 840 | 216 | 624 | 25.71% | 15,799 |
| off | on | 31.21% | 25.72 | 58,920 | 35 | 12 | 23 | 34.29% | 14,605 |
| on | off | 31.34% | 25.75 | 60,480 | 846 | 204 | 642 | 24.11% | 15,554 |
| on | on | 31.26% | 25.92 | 60,480 | 34 | 15 | 19 | 44.12% | 14,625 |

For the seeded predictor, adaptive throttling reduced prefetch attempts by 95.98%, wasted prefetches by 97.04%, and asynchronous copies by 5.97%. Decode throughput increased by 0.66%, while hit rate changed by -0.08 percentage points.

The seed adds 1,560 predictions in this short run, showing that prediction is available before the online matrix has learned its first transition. Prompt throughput was 820.05 tokens/s without transition collection and 816.92 tokens/s with collection in the fixed-prefetch rows, a 0.38% reduction. The single-case throughput and hit-rate differences are small and include run-to-run noise. A multi-case repeated benchmark is required before treating them as a general performance improvement.

## Limits and next measurements

- Transition collection currently runs in the Metal matrix-matrix prefill path. Very short prompts that stay on the matrix-vector path do not provide a prefill transition matrix.
- The 64 MiB arena stores a separate matrix for gate, up, and down tensors even though their route IDs match. Sharing one logical matrix per layer would reduce temporary profile memory by about 3x.
- The cost simulation covers victims required by capacity. The slot allocator can still evict extra entries for physical fragmentation recovery after admission.
- The remaining-token signal is optional. A future request-level backend hint would remove the environment variable from production integration.
- The ablation is a focused diagnostic, not the prior 10-input by 6-budget matrix. The next benchmark should repeat all four modes across multiple prompts and at least three seeds.
