#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import selectors
import signal
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("budget_eval", ROOT / "scripts/moe-cache-budget-eval.py")
BUDGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUDGET)
PREFIX = "GGML_METAL_MOE_CACHE_"
BASE = {"STRICT_UPDATE": "1", "LAYER_RESIDENCY": "1", "MM_ID": "1", "PREFETCH": "0",
        "ADAPTIVE_PREFETCH": "1", "PREFILL_MARKOV": "1", "PRELOAD": "1", "PRELOAD_REBUILD": "0",
        "BUNDLE": "1", "DYNAMIC_LAYER_BUDGET": "1", "PROTECTED_PERCENT": "80", "EVAL": "1"}
SHORT = "Explain why the sky is blue in one sentence."
PARAGRAPH = ("A database stores records in pages. A cache can keep frequently used pages in memory, "
             "while the remaining pages stay on disk. Measuring both latency and bytes transferred "
             "helps identify whether the bottleneck is computation, storage, or synchronization. ")
PROMPTS = {
    "short": SHORT,
    "prefill": "Explain the following design tradeoffs.\n" + PARAGRAPH * 3,
    "long": "Summarize the following engineering notes and explain two tradeoffs.\n" + PARAGRAPH * 24,
    "math": "A shop sells 16 items each day, keeps 3 for testing and donates 4. The remaining items sell for 2 dollars each. What is the daily revenue? Explain your calculation.",
    "code": "Write a Python function that maintains a fixed-capacity least-recently-used cache. Explain how the get and put operations update the data structure.",
    "chinese": "请解释为什么缓存命中率更高，不一定意味着程序运行速度更快。分别讨论读取成本、并发和同步开销，并给出一个具体例子。",
}


def cases(phase):
    result = []

    def add(name, budget, prompt, tokens, overrides=None):
        env = dict(BASE)
        env.update(overrides or {})
        env["REMAINING_TOKENS"] = str(tokens)
        result.append({"name": name, "budget_mib": budget, "prompt": prompt, "tokens": tokens, "env": env})

    if phase == "smoke":
        add("smoke", 1024, "short", 16)
    elif phase == "core":
        for budget in [256, 1024]:
            add("core-{}-base".format(budget), budget, "short", 128,
                {"STRICT_UPDATE": "0", "LAYER_RESIDENCY": "0", "MM_ID": "0", "PREFETCH": "1"})
            add("core-{}-update".format(budget), budget, "short", 128,
                {"LAYER_RESIDENCY": "0", "MM_ID": "0", "PREFETCH": "1"})
            add("core-{}-layer".format(budget), budget, "short", 128, {"MM_ID": "0", "PREFETCH": "1"})
    elif phase == "prefill":
        add("prefill-mv", 1024, "prefill", 64, {"MM_ID": "0"})
        add("prefill-mm", 1024, "prefill", 64)
    elif phase == "policies":
        variants = {"current": {}, "preload-off": {"PRELOAD": "0"}, "preload-rebuild": {"PRELOAD_REBUILD": "1"},
                    "layer-equal": {"DYNAMIC_LAYER_BUDGET": "0"}, "lru": {"PROTECTED_PERCENT": "0"},
                    "prefetch-adaptive": {"PREFETCH": "1"},
                    "prefetch-unthrottled": {"PREFETCH": "1", "ADAPTIVE_PREFETCH": "0"},
                    "no-prefill-markov": {"PREFETCH": "1", "PREFILL_MARKOV": "0"}, "no-bundle": {"BUNDLE": "0"}}
        for name, overrides in variants.items():
            add("policy-" + name, 1024, "long", 128, overrides)
    elif phase == "budget":
        for budget in [256, 512, 1024, 2048, 4096, 6144]:
            add("budget-{}".format(budget), budget, "short", 128)
    elif phase == "validation":
        for prompt in ["math", "code", "chinese"]:
            add("validation-" + prompt, 1024, prompt, 64)
            add("validation-" + prompt + "-base", 1024, prompt, 64,
                {"STRICT_UPDATE": "0", "LAYER_RESIDENCY": "0", "MM_ID": "0", "PREFETCH": "1"})
    else:
        raise ValueError("unknown phase " + phase)
    return result


def fingerprint(binary):
    paths = [binary, Path(__file__).resolve(), ROOT / "scripts/moe-cache-budget-eval.py"]
    paths += sorted(binary.parent.glob("libggml*.dylib")) + sorted(binary.parent.glob("libllama*.dylib"))
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(directory, model, binary, case, repetition):
    stem = "{}-r{}".format(case["name"], repetition)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_METAL_MOE_") and k != "GGML_MOE_ROUTE_TRACE"}
    env.update({PREFIX + k: v for k, v in case["env"].items()})
    prompt_file = directory / (case["prompt"] + "-prompt.txt")
    prompt_file.write_text(PROMPTS[case["prompt"]])
    argv = ["/usr/bin/time", "-l", str(binary), "-m", str(model), "--moe-cache-size", str(case["budget_mib"]),
            "-c", "4096", "-b", "64", "-ub", "64", "-t", "4", "-tb", "4", "-f", str(prompt_file),
            "-n", str(case["tokens"]), "--ignore-eos", "--seed", "42", "--temp", "0", "--no-display-prompt",
            "--no-conversation", "--no-warmup", "-fit", "off", "-v", "--log-file", str(directory / (stem + ".log")), "--log-colors", "off"]
    started = time.monotonic()
    first_output = None
    with (directory / (stem + "-output.txt")).open("wb") as output, (directory / (stem + "-time.txt")).open("wb") as timing:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=timing, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                if time.monotonic() - started > 600:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise TimeoutError(stem)
                for key, _ in selector.select(1):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    if first_output is None and data.strip():
                        first_output = time.monotonic() - started
                    output.write(data)
            status = process.wait(timeout=10)
        finally:
            selector.close()
            process.stdout.close()
    if status:
        raise RuntimeError("run failed: {} (exit {})".format(stem, status))
    row = BUDGET.parse_run(directory, stem)
    log = (directory / (stem + ".log")).read_text(errors="replace")
    iteration = [line.split("moe_cache_iteration:", 1)[1] for line in log.splitlines() if "moe_cache_iteration:" in line][-1]
    row.update({key.strip().replace(" ", "_"): int(value) for key, value in re.findall(r"([a-z ]+) = (\d+)", iteration)})
    prompt = re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?,\s*([\d.]+) tokens per second", log)[-1]
    row.update({"prompt_ms": float(prompt[0]), "prompt_tokens": int(prompt[1]), "prompt_tps": float(prompt[2]),
                "first_output_seconds": first_output, "wall_seconds": time.monotonic() - started,
                "stem": stem, "repetition": repetition, **case})
    for key in ["prefetch_attempts", "prefetch_admissions", "useful_prefetches", "wasted_prefetches", "cost_rejections", "preloaded_entries", "preloaded_bytes", "read_failures"]:
        label = {"preloaded_entries": "entries", "preloaded_bytes": "bytes"}.get(key, key.replace("_", " "))
        text = "\n".join(line for line in log.splitlines() if "ggml_metal_moe_cache_preload:" in line) if key.startswith("preloaded_") else log
        matches = re.findall(r"(?:^|, |: )" + re.escape(label) + r" = (\d+)", text)
        row[key] = int(matches[-1]) if matches else 0
    if row["steps"] != case["tokens"] - 1:
        raise ValueError("decode step mismatch in " + stem)
    expected = row["steps"] * 40 * 8 * (3 if case["env"]["BUNDLE"] == "0" else 1)
    if row["required"] != expected or row["sync_bundles"] != row["required"] - row["resident_hits"] or row["read_failures"]:
        raise ValueError("invalid cache counters in " + stem)
    return row


def report(directory, rows):
    summary = []
    for name in sorted({r["name"] for r in rows}):
        group = [r for r in rows if r["name"] == name]
        avg = lambda key: statistics.mean(r[key] for r in group)
        per_token = lambda key, divisor=1: statistics.mean(r[key] / r["steps"] / divisor for r in group)
        point = {"name": name, "n": len(group), "budget_mib": group[0]["budget_mib"], "prompt_tokens": avg("prompt_tokens"),
                 "decode_tps": avg("decode_tps"), "decode_sd": statistics.stdev(r["decode_tps"] for r in group) if len(group) > 1 else 0,
                 "prompt_tps": avg("prompt_tps"), "prompt_ms": avg("prompt_ms"), "first_output_seconds": avg("first_output_seconds"),
                 "footprint_mib": avg("footprint_mib"), "rss_mib": avg("rss_mib"), "max_footprint_mib": max(r["footprint_mib"] for r in group),
                 "hit_percent": statistics.mean(100 * r["resident_hits"] / r["required"] for r in group),
                 "loads_per_token": per_token("sync_bundles"), "read_mib_per_token": per_token("decode_read_bytes", 1024 ** 2),
                 "sync_mib_per_token": per_token("sync_bytes", 1024 ** 2), "pread_ms_per_token": per_token("decode_read_ns", 1e6),
                 "decode_ms_per_token": per_token("decode_ms"), "wait_calls_per_token": per_token("wait_calls"),
                 "wait_ms_per_token": per_token("wait_ns", 1e6), "gpu_ms_per_token": per_token("gpu_ns", 1e6),
                 "require_calls_per_token": per_token("require_calls"), "all_hit_calls_per_token": per_token("all_hit_calls"),
                 "disk_mib_per_token": per_token("disk_bytes", 1024 ** 2),
                 "prefill_top_recall_percent": statistics.mean(100 * r["prefill_top_hits"] / r["required"] for r in group),
                 "prefill_early_recall_percent": statistics.mean(100 * r["prefill_early_hits"] / r["early_required"] for r in group),
                 "output_hashes": sorted({r["output_sha256"] for r in group}), "env": group[0]["env"]}
        for key in ["prefetch_attempts", "prefetch_admissions", "useful_prefetches", "wasted_prefetches", "cost_rejections", "preloaded_entries", "preloaded_bytes"]:
            point[key] = avg(key)
        summary.append(point)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Shared-cache iteration evaluation", "", "Repeated runs; error is sample SD. One bundle is one layer/expert gate/up/down group except no-bundle cases.", "",
             "| Case | N | Decode tok/s | SD | Prompt tok/s | First output s | Hit % | Sync loads/token | Read MiB/token | Pread ms/token | Waits/token | Wait ms/token | GPU ms/token | Footprint MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in summary:
        lines.append("| {name} | {n} | {decode_tps:.2f} | {decode_sd:.2f} | {prompt_tps:.2f} | {first_output_seconds:.2f} | {hit_percent:.2f} | {loads_per_token:.2f} | {read_mib_per_token:.2f} | {pread_ms_per_token:.2f} | {wait_calls_per_token:.0f} | {wait_ms_per_token:.2f} | {gpu_ms_per_token:.2f} | {footprint_mib:.1f} |".format(**p))
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Reproducible shared-cache feature ablations on macOS")
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/llama-completion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "core", "prefill", "policies", "budget", "validation"], default="core")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records_file = directory / "records.json"
    rows = json.loads(records_file.read_text()) if records_file.exists() else []
    if args.analyze_only:
        report(directory, rows)
        return
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    manifest = {"model_path": str(args.model.resolve()), "git_commit": BUDGET.command(["git", "rev-parse", "HEAD"]),
                "fingerprints": fingerprint(args.binary.resolve()), "hardware": BUDGET.command(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"]),
                "model": BUDGET.read_gguf(args.model), "prompts": PROMPTS}
    manifest = json.loads(json.dumps(manifest))
    manifest_file = directory / "manifest.json"
    if rows:
        previous = json.loads(manifest_file.read_text())
        if previous != manifest:
            raise ValueError("configuration changed; use a new output directory")
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    schedule = [(case, repetition) for repetition in range(1, args.repetitions + 1) for case in cases(args.phase)]
    random.Random(42).shuffle(schedule)
    for case, repetition in schedule:
        matches = [r for r in rows if r["name"] == case["name"] and r["repetition"] == repetition]
        if matches:
            if any(matches[0][key] != value for key, value in case.items()):
                raise ValueError("case changed: " + case["name"])
            continue
        print("{} r{}".format(case["name"], repetition), flush=True)
        row = run(directory, args.model.resolve(), args.binary.resolve(), case, repetition)
        peers = [r for r in rows if r["name"] == case["name"]]
        if peers and row["output_sha256"] != peers[0]["output_sha256"]:
            raise ValueError("non-deterministic output: " + row["stem"])
        rows.append(row)
        records_file.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        report(directory, rows)
        print("  {:.2f} tok/s; {:.1f}% hit; {:.0f} waits/token".format(row["decode_tps"], row["resident_hits"] / row["required"] * 100, row["wait_calls"] / row["steps"]), flush=True)
    print(directory / "report.md", flush=True)


if __name__ == "__main__":
    main()
