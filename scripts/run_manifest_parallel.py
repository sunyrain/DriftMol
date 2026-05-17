#!/usr/bin/env python3
"""Run manifest entries in parallel across one or more CUDA devices."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "publication_ext" / "manifest.json"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    entries = data.get("entries", [])
    return entries if isinstance(entries, list) else []


def select_entries(
    entries: list[dict[str, Any]],
    groups: set[str] | None,
    names: set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    selected = entries
    if groups:
        selected = [entry for entry in selected if entry.get("group") in groups]
    if names:
        selected = [entry for entry in selected if entry.get("name") in names]
    if limit > 0:
        selected = selected[:limit]
    return selected


def complete(entry: dict[str, Any]) -> bool:
    return (ROOT / entry["output_dir"] / "final_metrics.json").exists()


def pop_next_incomplete(
    queue: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    force: bool = False,
) -> dict[str, Any] | None:
    """Pop the next still-incomplete entry, skipping outputs finished meanwhile."""
    while queue:
        entry = queue.pop(0)
        if not force and complete(entry):
            skipped.append(entry)
            continue
        return entry
    return None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = timestamp()
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_devices(raw: str | None) -> list[str]:
    if raw:
        devices = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = [item.strip() for item in visible.split(",") if item.strip()] if visible else ["0"]
    if not devices:
        raise ValueError("No CUDA devices selected")
    return devices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--group", action="append", help="Run only this group. Can be repeated.")
    parser.add_argument("--name", action="append", help="Run only this experiment name. Can be repeated.")
    parser.add_argument("--devices", help="Comma-separated CUDA device ids, e.g. 0,1,2,3.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--log-dir", default="outputs/publication_ext/parallel_logs")
    parser.add_argument("--status-file", default="outputs/publication_ext/parallel_runner_status.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    entries = select_entries(
        load_manifest(manifest_path),
        set(args.group) if args.group else None,
        set(args.name) if args.name else None,
        args.limit,
    )
    if not entries:
        print("No manifest entries selected.")
        return 0

    devices = parse_devices(args.devices)
    runnable = [entry for entry in entries if args.force or not complete(entry)]
    skipped = [entry for entry in entries if entry not in runnable]

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    status_file = Path(args.status_file)
    if not status_file.is_absolute():
        status_file = ROOT / status_file
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        assignments = [
            {
                "device": devices[idx % len(devices)],
                "name": entry["name"],
                "group": entry.get("group", ""),
                "command": entry["command"],
            }
            for idx, entry in enumerate(runnable)
        ]
        for item in assignments:
            print(f"[dry-run gpu{item['device']}] {item['group']} :: {item['name']}")
            print(f"  {item['command']}")
        if skipped:
            print(f"Skipped complete entries: {len(skipped)}")
        write_status(
            status_file,
            {
                "state": "dry_run",
                "manifest": rel(manifest_path),
                "devices": devices,
                "selected": len(entries),
                "runnable": len(runnable),
                "skipped_complete": [entry["name"] for entry in skipped],
                "assignments": assignments,
            },
        )
        return 0

    queue = list(runnable)
    running: dict[str, dict[str, Any]] = {}
    completed: list[str] = []
    failures: list[dict[str, Any]] = []

    def status_payload(state: str) -> dict[str, Any]:
        return {
            "state": state,
            "manifest": rel(manifest_path),
            "devices": devices,
            "selected": len(entries),
            "queued": [entry["name"] for entry in queue],
            "running": [
                {
                    "device": payload["device"],
                    "name": payload["entry"]["name"],
                    "group": payload["entry"].get("group", ""),
                    "log": rel(payload["log_path"]),
                }
                for payload in running.values()
            ],
            "completed": completed,
            "failures": failures,
            "skipped_complete": [entry["name"] for entry in skipped],
        }

    while queue or running:
        free_devices = [device for device in devices if device not in running]
        while queue and free_devices:
            device = free_devices.pop(0)
            entry = pop_next_incomplete(queue, skipped, args.force)
            if entry is None:
                break
            log_path = log_dir / f"{entry['name']}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            with log_path.open("a") as log:
                log.write(f"\n\n===== parallel runner start {timestamp()} device={device} =====\n")
                log.flush()
                proc = subprocess.Popen(
                    entry["command"],
                    cwd=ROOT,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
            running[device] = {"entry": entry, "proc": proc, "log_path": log_path, "device": device}
            print(f"[gpu{device}] started {entry['name']} log={rel(log_path)}")

        write_status(status_file, status_payload("running"))
        time.sleep(args.poll_seconds)

        for device, payload in list(running.items()):
            proc: subprocess.Popen = payload["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            entry = payload["entry"]
            if rc == 0:
                completed.append(entry["name"])
                print(f"[gpu{device}] completed {entry['name']}")
            else:
                failure = {
                    "name": entry["name"],
                    "group": entry.get("group", ""),
                    "returncode": rc,
                    "log": rel(payload["log_path"]),
                }
                failures.append(failure)
                print(f"[gpu{device}] FAILED {entry['name']} rc={rc} log={failure['log']}")
            del running[device]

    write_status(status_file, status_payload("failed" if failures else "completed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
