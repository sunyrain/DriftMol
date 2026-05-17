#!/usr/bin/env python3
"""Refresh reviewer-extra tables until all expected rows finish."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_complete() -> int:
    root = ROOT / "outputs" / "publication_ext" / "reviewer_extra"
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob("*/final_metrics.json"))


def run_collect() -> None:
    subprocess.run([sys.executable, "scripts/collect_reviewer_extra_results.py"], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--pid-file", default="outputs/publication_ext/reviewer_extra_postprocess.pid")
    args = parser.parse_args()

    pid_path = ROOT / args.pid_file
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n")

    print("[postprocess-reviewer-extra] started", flush=True)
    last = -1
    while True:
        complete = count_complete()
        print(f"[postprocess-reviewer-extra] complete={complete}/{args.expected}", flush=True)
        if complete != last:
            run_collect()
            last = complete
        if complete >= args.expected:
            break
        time.sleep(args.poll_seconds)
    run_collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
