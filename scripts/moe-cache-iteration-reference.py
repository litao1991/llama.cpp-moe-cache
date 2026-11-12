#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("iteration_eval", ROOT / "scripts/moe-cache-iteration-eval.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def main():
    parser = argparse.ArgumentParser(description="Full-model mapping references for the shared-cache experiment")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    binary = ROOT / "build/bin/llama-completion"
    rows = json.loads((directory / "records.json").read_text())
    references = []
    env = {key: value for key, value in os.environ.items() if not key.startswith("GGML_METAL_MOE_") and key != "GGML_MOE_ROUTE_TRACE"}
    schedule = [("short", 128)] * args.repetitions + [("prefill", 64), ("long", 128), ("math", 64)]
    for index, (prompt, tokens) in enumerate(schedule, 1):
        stem = "reference-{}-{}".format(prompt, index)
        prompt_path = directory / (prompt + "-prompt.txt")
        prompt_path.write_text(EVAL.PROMPTS[prompt])
        argv = ["/usr/bin/time", "-l", str(binary), "-m", str(args.model.resolve()), "--moe-cache-size", "0",
                "-c", "4096", "-b", "64", "-ub", "64", "-t", "4", "-tb", "4", "-f", str(prompt_path),
                "-n", str(tokens), "--ignore-eos", "--seed", "42", "--temp", "0", "--no-display-prompt", "--no-conversation",
                "--no-warmup", "-fit", "off", "-v", "--log-colors", "off", "--log-file", str(directory / (stem + ".log"))]
        print(stem, flush=True)
        with (directory / (stem + "-output.txt")).open("wb") as out, (directory / (stem + "-time.txt")).open("wb") as err:
            subprocess.run(argv, env=env, stdout=out, stderr=err, check=True, timeout=600, cwd=ROOT)
        log = (directory / (stem + ".log")).read_text(errors="replace")
        timing = (directory / (stem + "-time.txt")).read_text(errors="replace")
        milliseconds, steps, tps = re.findall(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs.*?,\s*([\d.]+) tokens per second", log)[-1]
        digest = hashlib.sha256((directory / (stem + "-output.txt")).read_bytes()).hexdigest()
        peers = [r for r in rows if r["prompt"] == prompt and r["env"]["MM_ID"] == "1"]
        references.append({"stem": stem, "prompt": prompt, "tokens": tokens, "decode_ms": float(milliseconds), "steps": int(steps),
                           "decode_tps": float(tps), "output_sha256": digest,
                           "matches_shared_mm": bool(peers) and all(r["output_sha256"] == digest for r in peers),
                           "footprint_mib": int(re.findall(r"(\d+)\s+peak memory footprint", timing)[-1]) / 1024 ** 2,
                           "rss_mib": int(re.findall(r"(\d+)\s+maximum resident set size", timing)[-1]) / 1024 ** 2})
        (directory / "reference-records.json").write_text(json.dumps(references, indent=2) + "\n")
    manifest = {"model_path": str(args.model.resolve()), "fingerprints": EVAL.fingerprint(binary),
                "reference_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / "reference-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory / "reference-records.json", flush=True)


if __name__ == "__main__":
    main()
