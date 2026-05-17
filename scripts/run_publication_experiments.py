#!/usr/bin/env python3
"""Run publication-stage experiments from configs/publication/manifest.json."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "publication" / "manifest.json"


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return list(data.get("entries", []))


def should_skip(entry: dict, force: bool) -> bool:
    if force:
        return False
    final_metrics = ROOT / entry["output_dir"] / "final_metrics.json"
    return final_metrics.exists()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--group", action="append", help="Run only this group. Can be repeated.")
    parser.add_argument("--name", action="append", help="Run only this experiment name. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument("--force", action="store_true", help="Run even if final_metrics.json exists.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of experiments to run.")
    parser.add_argument("--cuda-visible-devices", help="Set CUDA_VISIBLE_DEVICES for child training runs.")
    parser.add_argument(
        "--status-file",
        default="outputs/publication/runner_status.json",
        help="Status JSON path, relative to repo root unless absolute.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    entries = load_manifest(manifest_path)

    if args.group:
        allowed = set(args.group)
        entries = [e for e in entries if e["group"] in allowed]
    if args.name:
        allowed = set(args.name)
        entries = [e for e in entries if e["name"] in allowed]
    if args.limit > 0:
        entries = entries[: args.limit]

    if not entries:
        print("No manifest entries selected.")
        return 0

    log_dir = ROOT / "outputs" / "publication" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_file)
    if not status_path.is_absolute():
        status_path = ROOT / status_path
    run_env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    failures = []
    last_entry = None
    last_log = None
    for idx, entry in enumerate(entries, start=1):
        cmd = entry["command"]
        log_path = log_dir / f"{entry['name']}.log"
        last_entry = entry
        last_log = str(log_path.relative_to(ROOT))
        if should_skip(entry, args.force):
            print(f"[{idx}/{len(entries)}] skip complete: {entry['name']}")
            continue

        print(f"[{idx}/{len(entries)}] {entry['group']} :: {entry['name']}")
        print(f"  {cmd}")
        if args.dry_run:
            continue

        write_status(status_path, {
            "state": "running",
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "index": idx,
            "total": len(entries),
            "entry": entry,
            "log": str(log_path.relative_to(ROOT)),
            "cuda_visible_devices": run_env.get("CUDA_VISIBLE_DEVICES"),
        })
        with log_path.open("a") as log_file:
            log_file.write(
                f"\n\n===== runner start {dt.datetime.now(dt.timezone.utc).isoformat()} =====\n"
            )
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=run_env,
            )
        if proc.returncode != 0:
            failures.append((entry["name"], proc.returncode, str(log_path)))
            print(f"  FAILED rc={proc.returncode}; log={log_path.relative_to(ROOT)}")
            write_status(status_path, {
                "state": "failed",
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "index": idx,
                "total": len(entries),
                "entry": entry,
                "returncode": proc.returncode,
                "log": str(log_path.relative_to(ROOT)),
                "cuda_visible_devices": run_env.get("CUDA_VISIBLE_DEVICES"),
            })
        else:
            print(f"  done; log={log_path.relative_to(ROOT)}")
            write_status(status_path, {
                "state": "completed_entry",
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "index": idx,
                "total": len(entries),
                "entry": entry,
                "log": str(log_path.relative_to(ROOT)),
                "cuda_visible_devices": run_env.get("CUDA_VISIBLE_DEVICES"),
            })

    if failures:
        print("\nFailures:")
        for name, rc, log_path in failures:
            print(f"  {name}: rc={rc}, log={log_path}")
        return 1
    if args.dry_run:
        return 0
    write_status(status_path, {
        "state": "completed",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total": len(entries),
        "entry": last_entry or {},
        "log": last_log,
        "cuda_visible_devices": run_env.get("CUDA_VISIBLE_DEVICES"),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
