#!/usr/bin/env python3
"""Launch faithful_core after the destructive runner finishes successfully."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WATCH = ROOT / "outputs" / "publication_ext" / "parallel_runner_status.json"
DEFAULT_STATUS = ROOT / "outputs" / "reviewer_faithful" / "core_status.json"
DEFAULT_LOG_DIR = ROOT / "outputs" / "reviewer_faithful" / "logs"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def faithful_already_active_or_done(status_path: Path) -> bool:
    payload = read_json(status_path)
    return payload.get("state") in {"running", "completed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-status", default=str(DEFAULT_WATCH))
    parser.add_argument("--faithful-status", default=str(DEFAULT_STATUS))
    parser.add_argument("--devices", default="0,2,3")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--pid-file", default="")
    args = parser.parse_args()

    watch_status = Path(args.watch_status)
    if not watch_status.is_absolute():
        watch_status = ROOT / watch_status
    faithful_status = Path(args.faithful_status)
    if not faithful_status.is_absolute():
        faithful_status = ROOT / faithful_status
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    pid_file = Path(args.pid_file) if args.pid_file else None
    if pid_file is not None and not pid_file.is_absolute():
        pid_file = ROOT / pid_file
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()) + "\n")

    print(f"[defer] watching {rel(watch_status)}")
    print(f"[defer] faithful status {rel(faithful_status)}")
    print(f"[defer] target devices {args.devices}")
    sys.stdout.flush()

    while True:
        if faithful_already_active_or_done(faithful_status):
            print("[defer] faithful_core already running or completed; exiting")
            return 0

        payload = read_json(watch_status)
        state = payload.get("state")
        running = payload.get("running", [])
        queued = payload.get("queued", [])
        failures = payload.get("failures", [])

        if state == "completed":
            print("[defer] destructive runner completed; collecting extension results")
            subprocess.run(
                [sys.executable, "scripts/collect_extension_results.py"],
                cwd=ROOT,
                check=False,
            )
            cmd = [
                sys.executable,
                "scripts/run_manifest_parallel.py",
                "--manifest",
                "configs/reviewer_faithful/manifest.json",
                "--group",
                "faithful_core",
                "--devices",
                args.devices,
                "--poll-seconds",
                "30",
                "--status-file",
                rel(faithful_status),
                "--log-dir",
                rel(log_dir),
            ]
            print("[defer] launching: " + " ".join(cmd))
            sys.stdout.flush()
            rc = subprocess.call(cmd, cwd=ROOT)
            print(f"[defer] faithful_core runner exited with code {rc}; refreshing artifacts")
            sys.stdout.flush()
            for refresh_cmd in [
                [sys.executable, "scripts/collect_faithful_drifting_results.py"],
                [sys.executable, "scripts/audit_drifting_faithfulness.py"],
                [sys.executable, "scripts/audit_reviewer_experiment_readiness.py"],
            ]:
                subprocess.run(refresh_cmd, cwd=ROOT, check=False)
            return rc

        if state == "failed" or failures:
            print(f"[defer] destructive runner failed; not launching faithful_core. failures={failures}")
            return 2

        print(
            f"[defer] waiting: state={state!r}, running={len(running)}, queued={len(queued)}"
        )
        sys.stdout.flush()
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
