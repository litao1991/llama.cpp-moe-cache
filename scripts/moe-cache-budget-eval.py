#!/usr/bin/env python3

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import struct
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
MIB = 1024 ** 2


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def read_gguf(path):
    primitive = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    sizes = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2)}
    with path.open("rb") as source:
        def number(fmt):
            return struct.unpack("<" + fmt, source.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return source.read(number("Q")).decode("utf-8")

        def value(kind, keep=True):
            if kind in primitive:
                return number(primitive[kind])
            if kind == 8:
                length = number("Q")
                if keep:
                    return source.read(length).decode("utf-8")
                source.seek(length, 1)
                return None
            if kind == 9:
                subtype, count = number("I"), number("Q")
                if subtype in primitive:
                    source.seek(count * struct.calcsize("<" + primitive[subtype]), 1)
                else:
                    for _ in range(count):
                        value(subtype, False)
                return None
            raise ValueError("unknown GGUF metadata type {}".format(kind))

        if source.read(4) != b"GGUF" or number("I") != 3:
            raise ValueError("GGUF v3 little-endian model required")
        tensor_count, field_count = number("Q"), number("Q")
        metadata = {}
        for _ in range(field_count):
            key, kind = string(), number("I")
            metadata[key] = value(kind)
        tensors = []
        for _ in range(tensor_count):
            name, dimensions = string(), number("I")
            shape = [number("Q") for _ in range(dimensions)]
            kind, offset = number("I"), number("Q")
            if kind not in sizes:
                raise ValueError("unsupported tensor type {} in {}".format(kind, name))
            block, block_bytes = sizes[kind]
            tensors.append({"name": name, "shape": shape, "offset": offset, "bytes": math.prod(shape) // block * block_bytes})
        alignment = metadata.get("general.alignment", 32)
        data_offset = (source.tell() + alignment - 1) // alignment * alignment

    arch = metadata["general.architecture"]
    layers = metadata[arch + ".block_count"] - metadata.get(arch + ".nextn_predict_layers", 0)
    experts = metadata[arch + ".expert_count"]
    top_k = metadata[arch + ".expert_used_count"]
    groups = {}
    dense_bytes = 0
    for tensor in tensors:
        match = re.match(r"blk\.(\d+)\.", tensor["name"])
        if match and int(match.group(1)) >= layers:
            continue
        if re.search(r"\.ffn_(gate|up|down)_exps\.weight$", tensor["name"]):
            groups.setdefault(int(match.group(1)), []).append(tensor)
        elif tensor["name"] != "token_embd.weight":
            dense_bytes += tensor["bytes"]
    expert_bytes = sum(t["bytes"] for members in groups.values() for t in members)
    return {"layers": layers, "experts": experts, "top_k": top_k, "dense_weight_bytes": dense_bytes,
            "expert_weight_bytes": expert_bytes, "active_expert_bytes": expert_bytes * top_k / experts,
            "active_weight_bytes": dense_bytes + expert_bytes * top_k / experts,
            "mean_bundle_bytes": expert_bytes / experts / len(groups), "data_offset": data_offset,
            "expert_groups": groups, "file_bytes": path.stat().st_size}


class Rusage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 18)]


def disk_bytes():
    info = Rusage()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    result = lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info))
    return int(info.values[16]) if result == 0 else None


def calibrate_io(path, model):
    samples = []
    rng = random.Random(42)
    for cached in [True, False]:
        fd = os.open(path, os.O_RDONLY)
        try:
            if not cached:
                fcntl.fcntl(fd, 48, 1)
                fcntl.fcntl(fd, 45, 0)
            requests = []
            for _ in range(128):
                members = model["expert_groups"][rng.choice(sorted(model["expert_groups"]))]
                expert = rng.randrange(model["experts"])
                requests.append([(model["data_offset"] + t["offset"] + expert * (t["bytes"] // model["experts"]), t["bytes"] // model["experts"]) for t in members])
            start_disk = disk_bytes()
            latencies, total, bundle_samples = [], 0, []
            started = time.perf_counter()
            for request in requests:
                before = disk_bytes()
                tick = time.perf_counter()
                request_bytes = 0
                for offset, size in request:
                    data = os.pread(fd, size, offset)
                    if len(data) != size:
                        raise ValueError("short calibration read")
                    total += size
                    request_bytes += size
                latency = time.perf_counter() - tick
                after = disk_bytes()
                latencies.append(latency)
                bundle_samples.append({"bytes": request_bytes, "seconds": latency, "disk_bytes": None if before is None or after is None else after - before})
            elapsed = time.perf_counter() - started
            end_disk = disk_bytes()
            storage = [s for s in bundle_samples if s["disk_bytes"] is not None and s["disk_bytes"] >= s["bytes"] * 0.9]
            memory = [s for s in bundle_samples if s["disk_bytes"] == 0]
            samples.append({"mode": "buffered" if cached else "F_NOCACHE_requested", "bytes": total, "seconds": elapsed,
                            "bundle_mean_ms": statistics.mean(latencies) * 1000, "bundle_median_ms": statistics.median(latencies) * 1000,
                            "bandwidth_GBps": total / elapsed / 1e9, "process_disk_bytes": None if start_disk is None or end_disk is None else end_disk - start_disk,
                            "storage_dominated_count": len(storage), "storage_dominated_mean_ms": statistics.mean(s["seconds"] for s in storage) * 1000 if storage else None,
                            "storage_dominated_GBps": sum(s["bytes"] for s in storage) / sum(s["seconds"] for s in storage) / 1e9 if storage else None,
                            "zero_disk_count": len(memory), "zero_disk_mean_ms": statistics.mean(s["seconds"] for s in memory) * 1000 if memory else None,
                            "bundle_samples": bundle_samples})
        finally:
            os.close(fd)
    return samples


def parse_run(directory, stem):
    text = (directory / (stem + ".log")).read_text(errors="replace")
    timing = (directory / (stem + "-time.txt")).read_text(errors="replace")
    lines = [line for line in text.splitlines() if "moe_cache_eval: steps =" in line]
    if not lines:
        raise ValueError("missing instrumented decode metrics")
    metrics = {key.strip().replace(" ", "_"): int(value) for key, value in re.findall(r"([a-z ]+) = (\d+)", lines[-1].split("moe_cache_eval:", 1)[1])}
    evals = re.findall(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs.*?,\s*([\d.]+) tokens per second", text)
    if not evals:
        raise ValueError("missing decode time")
    milliseconds, steps, tps = map(float, evals[-1])
    if metrics["steps"] != int(steps):
        raise ValueError("metric step mismatch {} != {}".format(metrics["steps"], steps))
    metrics.update({"decode_ms": milliseconds, "decode_tps": tps,
                    "rss_mib": int(re.findall(r"(\d+)\s+maximum resident set size", timing)[-1]) / MIB,
                    "footprint_mib": int(re.findall(r"(\d+)\s+peak memory footprint", timing)[-1]) / MIB,
                    "output_sha256": hashlib.sha256((directory / (stem + "-output.txt")).read_bytes()).hexdigest()})
    return metrics


def write_report(directory, records, model, bandwidth):
    manifest_file = directory / "manifest.json"
    calibration = json.loads(manifest_file.read_text()).get("io_calibration", []) if manifest_file.exists() else []
    samples = [sample for group in calibration for sample in group.get("bundle_samples", [])]
    storage = [s for s in samples if s["disk_bytes"] is not None and s["disk_bytes"] >= s["bytes"] * 0.9]
    memory = [s for s in samples if s["disk_bytes"] == 0]
    storage_bw = sum(s["bytes"] for s in storage) / sum(s["seconds"] for s in storage) if storage else None
    memory_bw = sum(s["bytes"] for s in memory) / sum(s["seconds"] for s in memory) if memory else None
    reference = []
    for log in sorted(directory.glob("resident-r*.log")):
        matches = re.findall(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs.*?,\s*([\d.]+) tokens per second", log.read_text(errors="replace"))
        if matches:
            milliseconds, steps, tps = map(float, matches[-1])
            reference.append({"decode_ms_per_token": milliseconds / steps, "decode_tps": tps})
    lines = ["# Shared-cache P0 budget evaluation", "", "Memory bandwidth assumption: {:.0f} GB/s (Apple M5 specification).".format(bandwidth), "",
             "| Requested MiB | Actual MiB | Decode mean +/- sd tok/s | Before-load hit % | Sync bundles/token | Read MiB/token | Pread ms/token | Total ms/token | Footprint MiB |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = []
    for budget in sorted({row["budget_mib"] for row in records}):
        rows = [r for r in records if r["budget_mib"] == budget]
        avg = lambda key: statistics.mean(row[key] for row in rows)
        hit = statistics.mean(r["resident_hits"] / r["required"] for r in rows)
        loads = statistics.mean(r["sync_bundles"] / r["steps"] for r in rows)
        read = statistics.mean(r["decode_read_bytes"] / r["steps"] for r in rows)
        pread_ms = statistics.mean(r["decode_read_ns"] / r["steps"] / 1e6 for r in rows)
        total_ms = statistics.mean(r["decode_ms"] / r["steps"] for r in rows)
        memory_ms = (model["active_weight_bytes"] + 2 * read) / (bandwidth * 1e9) * 1000
        point = {"budget_mib": budget, "actual_mib": avg("capacity") / MIB, "tps_mean": avg("decode_tps"),
                 "tps_sd": statistics.stdev(r["decode_tps"] for r in rows) if len(rows) > 1 else 0,
                 "hit_fraction": hit, "sync_bundles_per_token": loads, "read_bytes_per_token": read,
                 "pread_ms_per_token": pread_ms, "total_ms_per_token": total_ms,
                 "residual_ms_per_token": total_ms - pread_ms, "footprint_mib": avg("footprint_mib"),
                 "dram_floor_ms": memory_ms, "dram_roof_tps": 1000 / memory_ms,
                 "process_disk_bytes_per_token": statistics.mean(r["disk_bytes"] / r["steps"] for r in rows if r["disk_available"]) if any(r["disk_available"] for r in rows) else None}
        point["sync_bytes_per_token"] = statistics.mean(r["sync_bytes"] / r["steps"] for r in rows)
        point["extra_read_bytes_per_token"] = read - point["sync_bytes_per_token"]
        if reference:
            point["reference_ms_per_token"] = statistics.mean(r["decode_ms_per_token"] for r in reference)
            point["extra_nonread_ms"] = point["residual_ms_per_token"] - point["reference_ms_per_token"]
        if storage_bw and memory_bw and point["process_disk_bytes_per_token"] is not None:
            disk = min(read, point["process_disk_bytes_per_token"])
            point["calibrated_read_ms"] = (disk / storage_bw + (read - disk) / memory_bw) * 1000
            if reference:
                point["reference_plus_io_ms"] = point["reference_ms_per_token"] + point["calibrated_read_ms"]
                point["reference_plus_io_tps"] = 1000 / point["reference_plus_io_ms"]
        summary.append(point)
        lines.append("| {budget_mib} | {actual_mib:.0f} | {tps_mean:.2f} +/- {tps_sd:.2f} | {hit:.2f} | {sync_bundles_per_token:.2f} | {read:.2f} | {pread_ms_per_token:.2f} | {total_ms_per_token:.2f} | {footprint_mib:.1f} |".format(**point, hit=hit * 100, read=read / MIB))
    lines += ["", "Each point averages repeated runs; all runs use one fixed prompt. Process disk bytes are OS-attributed process storage reads, not expert-specific SSD misses.", "",
              "DRAM-only floor = (estimated active weight bytes + 2 * logical file-read bytes/token) / specified bandwidth. This optimistic traffic model excludes quantization, graph synchronization, KV and scheduling costs.", "",
              "Raw data: records.json; model/hardware: manifest.json; per-run logs and time output are retained."]
    if storage_bw and memory_bw:
        lines += ["", "Application-level I/O calibration: {:.2f} GB/s for {} storage-dominated samples; {:.2f} GB/s for {} zero-disk samples. These are not device peak specifications.".format(storage_bw / 1e9, len(storage), memory_bw / 1e9, len(memory))]
    if reference:
        lines += ["", "Full-model mapping reference: {} tok/s; mean {:.2f} tok/s.".format([r["decode_tps"] for r in reference], statistics.mean(r["decode_tps"] for r in reference))]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    width, height = 840, 340
    ymax = max(p["tps_mean"] + p["tps_sd"] for p in summary) * 1.25
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="840" height="340" viewBox="0 0 840 340">', '<rect width="840" height="340" fill="white"/>', '<text x="420" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Shared-cache decode throughput (mean and sample SD)</text>']
    for i, point in enumerate(summary):
        x = 55 + i * 750 / len(summary)
        h = point["tps_mean"] / ymax * 225
        error = point["tps_sd"] / ymax * 225
        y = 275 - h
        svg += ['<rect x="{:.1f}" y="{:.1f}" width="70" height="{:.1f}" fill="#0f766e"/>'.format(x, y, h), '<path d="M {0:.1f} {1:.1f} V {2:.1f}" stroke="black" stroke-width="2"/>'.format(x + 35, y - error, y + error), '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" font-family="sans-serif">{:.2f}</text>'.format(x + 35, y - error - 9, point["tps_mean"]), '<text x="{:.1f}" y="301" text-anchor="middle" font-family="sans-serif">{}</text>'.format(x + 35, point["budget_mib"])]
    svg += ['<text x="420" y="329" text-anchor="middle" font-family="sans-serif">Requested expert-cache budget (MiB); throughput in tok/s</text>', '</svg>']
    (directory / "decode-budget.svg").write_text("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Measure shared MoE cache budget and decode costs on macOS")
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/llama-completion")
    parser.add_argument("--output", type=Path, default=ROOT / "moe-cache-eval/budget-20261112")
    parser.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096, 6144])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--skip-io-calibration", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records_file = args.output / "records.json"
    if args.calibrate_only:
        manifest_file = args.output / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["io_calibration"] = calibrate_io(args.model, manifest["model"])
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        return
    if args.analyze_only:
        manifest = json.loads((args.output / "manifest.json").read_text())
        write_report(args.output, json.loads(records_file.read_text()), manifest["model"], 153)
        return
    model = read_gguf(args.model)
    manifest = {"model_path": str(args.model.resolve()), "model": model, "budgets_mib": args.budgets,
                "tokens": args.tokens, "repetitions": args.repetitions, "seed": 42,
                "git_commit": command(["git", "rev-parse", "HEAD"]),
                "diff_sha256": hashlib.sha256(command(["git", "diff"]).encode()).hexdigest(),
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "memory_bandwidth_GBps": 153, "hardware": command(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu"]),
                "macos": command(["sw_vers", "-productVersion"]), "io_calibration": []}
    records = json.loads(records_file.read_text()) if records_file.exists() else []
    manifest_file = args.output / "manifest.json"
    if records and manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        for key in ["model_path", "tokens", "seed", "git_commit", "diff_sha256", "binary_sha256"]:
            if previous.get(key) != manifest.get(key):
                raise ValueError("resume configuration changed: {}".format(key))
        manifest["io_calibration"] = previous.get("io_calibration", [])
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    schedule = [(budget, repetition) for repetition in range(1, args.repetitions + 1) for budget in args.budgets]
    random.Random(42).shuffle(schedule)
    for index, (budget, repetition) in enumerate(schedule):
        if any(r["budget_mib"] == budget and r["repetition"] == repetition for r in records):
            continue
        stem = "{}m-r{}".format(budget, repetition)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_METAL_MOE_") and k != "GGML_MOE_ROUTE_TRACE"}
        env["GGML_METAL_MOE_CACHE_EVAL"] = "1"
        argv = ["/usr/bin/time", "-l", str(args.binary.resolve()), "-m", str(args.model.resolve()),
                "--moe-cache-size", str(budget), "-c", "2048", "-b", "64", "-ub", "64", "-t", "4", "-tb", "4",
                "-p", "Explain why the sky is blue in one sentence.", "-n", str(args.tokens), "--ignore-eos", "--seed", "42", "--temp", "0",
                "--no-display-prompt", "--no-conversation", "--no-warmup", "-fit", "off", "-v", "--log-file", str((args.output / (stem + ".log")).resolve()), "--log-colors", "off"]
        print("[{}/{}] {}".format(index + 1, len(schedule), stem), flush=True)
        with (args.output / (stem + "-output.txt")).open("w") as out, (args.output / (stem + "-time.txt")).open("w") as err:
            completed = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err, timeout=600)
        if completed.returncode:
            raise RuntimeError("run failed: {}".format(stem))
        row = parse_run(args.output, stem)
        if row["steps"] != args.tokens - 1 or row["required"] != row["steps"] * model["layers"] * model["top_k"]:
            raise ValueError("unexpected routed demand count")
        if records and row["output_sha256"] != records[0]["output_sha256"]:
            raise ValueError("generated output mismatch: {}".format(stem))
        row.update({"budget_mib": budget, "repetition": repetition, "stem": stem})
        records.append(row)
        records_file.write_text(json.dumps(records, indent=2) + "\n")
        write_report(args.output, records, model, 153)
        print("  {:.2f} tok/s, hit {:.1f}%, {:.1f} sync bundles/token".format(row["decode_tps"], 100 * row["resident_hits"] / row["required"], row["sync_bundles"] / row["steps"]), flush=True)
    if not args.skip_io_calibration:
        manifest["io_calibration"] = calibrate_io(args.model, model)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_report(args.output, records, model, 153)
    print(args.output / "report.md", flush=True)


if __name__ == "__main__":
    main()
