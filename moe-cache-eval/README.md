# Archived shared-cache evaluation results

The tracked files in this directory are compact experiment records, summaries, configuration fingerprints, charts, and selected test logs.

- `budget-20261112`: 18 original shared-P0 budget runs. See [the budget report](../docs/development/moe-shared-cache-budget-m5-evaluation-zh.md).
- `iteration-20261112-v1`: 64 feature/budget/regression runs and 6 full-model mapping references. See [the iteration report](../docs/development/moe-shared-cache-iteration-plan-and-evaluation-zh.md).

The manifests retain the baseline commit and runtime fingerprints used when measuring. They are not the archive commit IDs. No model weights, executables, build directories, or per-inference verbose logs are included in this archive. References to those local logs in experiment reports describe the original working environment.

To generate a fresh complete set of logs and results on macOS:

```bash
bash scripts/moe-cache-iteration-eval.sh /absolute/path/to/model.gguf moe-cache-eval/iteration-new-run
```

Run the command from the repository root. Use a new output directory after changing code, binaries, or evaluation scripts.
