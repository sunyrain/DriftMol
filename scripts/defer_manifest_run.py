#!/usr/bin/env python3
"""Wait for prerequisite files, then run selected manifest entries."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def missing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--wait-for", action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--pid-file")
    parser.add_argument("--label", default="deferred-manifest-run")
    args = parser.parse_args()

    if args.pid_file:
        pid_path = ROOT / args.pid_file
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(f"{os.getpid()}\n")

    wait_paths = [ROOT / path for path in args.wait_for]
    print(f"[{args.label}] started at {timestamp()}", flush=True)
    print(f"[{args.label}] names={','.join(args.name)} devices={args.devices}", flush=True)
    if wait_paths:
        print(f"[{args.label}] waiting for {len(wait_paths)} prerequisite files", flush=True)

    while True:
        missing_paths = missing(wait_paths)
        if not missing_paths:
            break
        done = existing(wait_paths)
        print(
            f"[{args.label}] ready={len(done)}/{len(wait_paths)} "
            f"missing={','.join(rel(path) for path in missing_paths[:4])}",
            flush=True,
        )
        time.sleep(args.poll_seconds)

    cmd = [
        sys.executable,
        "scripts/run_manifest_parallel.py",
        "--manifest",
        args.manifest,
        "--devices",
        args.devices,
        "--poll-seconds",
        "30",
        "--status-file",
        args.status_file,
        "--log-dir",
        args.log_dir,
    ]
    for name in args.name:
        cmd.extend(["--name", name])

    print(f"[{args.label}] prerequisites ready; launching at {timestamp()}", flush=True)
    print(f"[{args.label}] command={' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"[{args.label}] finished rc={proc.returncode} at {timestamp()}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
