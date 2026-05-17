#!/usr/bin/env python3
"""Refresh generalization result tables while deferred runs finish."""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_complete(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return sum(1 for _ in output_root.glob("*/final_metrics.json"))


def run_collect() -> None:
    subprocess.run(
        ["python", "scripts/collect_generalization_results.py"],
        cwd=ROOT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--pid-file", default="outputs/publication_ext/generalization_postprocess.pid")
    parser.add_argument("--output-root", default="outputs/publication_ext/generalization")
    args = parser.parse_args()

    pid_file = ROOT / args.pid_file
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")

    output_root = ROOT / args.output_root
    print("[postprocess-generalization] started", flush=True)
    last = -1
    while True:
        count = count_complete(output_root)
        print(f"[postprocess-generalization] complete={count}/{args.expected}", flush=True)
        if count != last:
            run_collect()
            last = count
        if count >= args.expected:
            break
        time.sleep(args.poll_seconds)
    run_collect()


if __name__ == "__main__":
    main()
