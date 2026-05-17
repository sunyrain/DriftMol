#!/usr/bin/env python3
"""Summarize extension experiment status, runner state, and GPU use."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_EXT = ROOT / "outputs" / "publication_ext"
OUT_REVIEWER = ROOT / "outputs" / "reviewer_faithful"
RESULTS = ROOT / "results"
EPOCH_RE = re.compile(r"\[epoch\s+(?P<epoch>\d+)\]")
TRAIN_RE = re.compile(r"\[train\]\s+(?P<epochs>\d+)\s+epochs")
DURATION_RE = re.compile(r"\((?P<seconds>\d+)s\)")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def status_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    running = []
    done_log_count = 0
    for item in payload.get("running", []):
        enriched = dict(item)
        log = ROOT / str(item.get("log", ""))
        last_epoch, total_epochs, signal = parse_log(log)
        log_done = signal.startswith("[done]")
        if log_done:
            enriched["progress"] = "final"
            enriched["eta"] = "final"
            done_log_count += 1
        elif last_epoch is not None and total_epochs is not None:
            enriched["progress"] = f"{last_epoch}/{total_epochs}"
            enriched["eta"] = format_eta(last_epoch, total_epochs, signal)
        elif last_epoch is not None:
            enriched["progress"] = str(last_epoch)
            enriched["eta"] = "-"
        else:
            enriched["progress"] = "-"
            enriched["eta"] = "-"
        enriched["last_signal"] = signal
        running.append(enriched)
    state = payload.get("state", "-")
    if state == "running" and running and done_log_count == len(running):
        state = "stale_done"
    return {
        "path": rel(path),
        "state": state,
        "selected": payload.get("selected", "-"),
        "runnable": payload.get("runnable", "-"),
        "devices": payload.get("devices", []),
        "running": running,
        "completed": payload.get("completed", []),
        "failures": payload.get("failures", []),
        "updated_at": payload.get("updated_at", ""),
    }


def parse_log(log_path: Path) -> tuple[int | None, int | None, str]:
    if not log_path.exists():
        return None, None, "missing_log"
    last_epoch = None
    total_epochs = None
    last_signal = ""
    for line in log_path.read_text(errors="replace").splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            total_epochs = int(train_match.group("epochs"))
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            last_epoch = int(epoch_match.group("epoch"))
            last_signal = line.strip()
        elif line.startswith("[done]") or line.startswith("[final]"):
            last_signal = line.strip()
    return last_epoch, total_epochs, last_signal


def format_eta(last_epoch: int, total_epochs: int, signal: str) -> str:
    if last_epoch >= total_epochs:
        return "final"
    match = DURATION_RE.search(signal)
    if not match:
        return "-"
    seconds = max(0, (total_epochs - last_epoch) * int(match.group("seconds")))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def result_status(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "path": rel(path),
        "num_experiments": payload.get("num_experiments", 0),
        "complete": payload.get("complete", 0),
        "pending_or_incomplete": payload.get("pending_or_incomplete", 0),
        "minimum_reached": payload.get("minimum_completed_runs_reached"),
        "status_counts": payload.get("status_counts", {}),
    }


def faithful_status(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "path": rel(path),
        "num_experiments": payload.get("num_experiments", 0),
        "complete": payload.get("complete", 0),
        "pending_or_incomplete": payload.get("pending_or_incomplete", 0),
        "faithful_core_complete": payload.get("faithful_core_complete", False),
        "groups": payload.get("groups", {}),
    }


def live_snapshot(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    return {
        "path": rel(path),
        "note": payload.get("note", ""),
        "rows": rows,
    }


def refresh_vae_drift_live_snapshot() -> None:
    try:
        from scripts.collect_vae_drift_live_snapshot import (
            DEFAULT_LOG_DIR,
            DEFAULT_MANIFEST,
            OUT_JSON,
            OUT_TEX,
            build_snapshot,
            write_tex,
        )
    except Exception:
        return
    try:
        snapshot = build_snapshot(DEFAULT_MANIFEST, DEFAULT_LOG_DIR)
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(snapshot, indent=2) + "\n")
        write_tex(OUT_TEX, snapshot)
    except Exception:
        return


def pid_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "alive": False, "pid": None}
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        return {"path": rel(path), "exists": True, "alive": False, "pid": None}
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = True
    return {"path": rel(path), "exists": True, "alive": alive, "pid": pid}


def gpu_status() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    except Exception:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append({
            "index": parts[0],
            "name": parts[1],
            "memory_used_mb": parts[2],
            "memory_total_mb": parts[3],
            "utilization_gpu_pct": parts[4],
        })
    return rows


def disk_status(path: Path = ROOT) -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["df", "-h", str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return {}
    lines = proc.stdout.splitlines()
    if len(lines) < 2:
        return {}
    parts = lines[1].split()
    if len(parts) < 6:
        return {}
    return {
        "filesystem": parts[0],
        "size": parts[1],
        "used": parts[2],
        "available": parts[3],
        "use_percent": parts[4],
        "mounted_on": parts[5],
    }


def build_payload() -> dict[str, Any]:
    refresh_vae_drift_live_snapshot()
    runner_statuses = [status_summary(path) for path in sorted(OUT_EXT.glob("parallel_runner_status*.json"))]
    reviewer_statuses = [
        status_summary(path)
        for path in sorted(OUT_REVIEWER.glob("*status.json"))
    ]
    return {
        "extension_completion": read_json(RESULTS / "extension_completion_status.json"),
        "destructive": result_status(RESULTS / "destructive_ablation_status.json"),
        "vae_sensitivity": result_status(RESULTS / "vae_sensitivity_status.json"),
        "vae_drift": result_status(RESULTS / "vae_drift_downstream_status.json"),
        "trained_baselines": result_status(RESULTS / "trained_baseline_status.json"),
        "generalization": result_status(RESULTS / "generalization_status.json"),
        "reviewer_extra": result_status(RESULTS / "reviewer_extra_status.json"),
        "next_wave": result_status(RESULTS / "next_wave_status.json"),
        "vae_drift_live_snapshot": live_snapshot(
            RESULTS / "vae_drift_live_snapshot.json"
        ),
        "faithful_drifting": faithful_status(RESULTS / "faithful_drifting_status.json"),
        "runner_statuses": runner_statuses,
        "reviewer_runner_statuses": reviewer_statuses,
        "deferred_faithful_launcher": pid_status(
            OUT_REVIEWER / "deferred_faithful_core_launcher.pid"
        ),
        "deferred_vae_drift_launchers": [
            pid_status(path)
            for path in sorted(OUT_EXT.glob("vae_drift_launcher_*.pid"))
        ],
        "deferred_generalization_launchers": [
            pid_status(path)
            for path in sorted(OUT_EXT.glob("generalization_launcher_*.pid"))
        ],
        "deferred_reviewer_extra_launchers": [
            pid_status(path)
            for path in sorted(OUT_EXT.glob("reviewer_extra_launcher_*.pid"))
        ],
        "vae_drift_postprocess": pid_status(
            OUT_EXT / "vae_drift_postprocess.pid"
        ),
        "generalization_postprocess": pid_status(
            OUT_EXT / "generalization_postprocess.pid"
        ),
        "reviewer_extra_postprocess": pid_status(
            OUT_EXT / "reviewer_extra_postprocess.pid"
        ),
        "gpus": gpu_status(),
        "disk": disk_status(ROOT),
    }


def print_table(payload: dict[str, Any]) -> None:
    ext = payload.get("extension_completion", {})
    print(f"Extension complete: {ext.get('complete')}")
    for key in (
        "destructive",
        "vae_sensitivity",
        "vae_drift",
        "trained_baselines",
        "generalization",
        "reviewer_extra",
        "next_wave",
    ):
        status = payload[key]
        print(
            f"{key}: complete={status['complete']}/{status['num_experiments']} "
            f"pending={status['pending_or_incomplete']} minimum_reached={status['minimum_reached']} "
            f"status_counts={status['status_counts']}"
        )
    faithful = payload["faithful_drifting"]
    core = faithful.get("groups", {}).get("faithful_core", {})
    alloc = faithful.get("groups", {}).get("faithful_allocation", {})
    print(
        "faithful_drifting: "
        f"complete={faithful['complete']}/{faithful['num_experiments']} "
        f"pending={faithful['pending_or_incomplete']} "
        f"core={core.get('complete', 0)}/{core.get('total', 0)} "
        f"allocation={alloc.get('complete', 0)}/{alloc.get('total', 0)} "
        f"core_complete={faithful['faithful_core_complete']}"
    )
    launcher = payload["deferred_faithful_launcher"]
    print(
        "deferred_faithful_launcher: "
        f"pid={launcher.get('pid')} alive={launcher.get('alive')} path={launcher.get('path')}"
    )
    for launcher in payload.get("deferred_vae_drift_launchers", []):
        print(
            "deferred_vae_drift_launcher: "
            f"pid={launcher.get('pid')} alive={launcher.get('alive')} path={launcher.get('path')}"
        )
    for launcher in payload.get("deferred_generalization_launchers", []):
        print(
            "deferred_generalization_launcher: "
            f"pid={launcher.get('pid')} alive={launcher.get('alive')} path={launcher.get('path')}"
        )
    for launcher in payload.get("deferred_reviewer_extra_launchers", []):
        print(
            "deferred_reviewer_extra_launcher: "
            f"pid={launcher.get('pid')} alive={launcher.get('alive')} path={launcher.get('path')}"
        )
    postprocess = payload.get("vae_drift_postprocess", {})
    print(
        "vae_drift_postprocess: "
        f"pid={postprocess.get('pid')} alive={postprocess.get('alive')} path={postprocess.get('path')}"
    )
    gen_postprocess = payload.get("generalization_postprocess", {})
    print(
        "generalization_postprocess: "
        f"pid={gen_postprocess.get('pid')} alive={gen_postprocess.get('alive')} path={gen_postprocess.get('path')}"
    )
    extra_postprocess = payload.get("reviewer_extra_postprocess", {})
    print(
        "reviewer_extra_postprocess: "
        f"pid={extra_postprocess.get('pid')} alive={extra_postprocess.get('alive')} path={extra_postprocess.get('path')}"
    )

    snapshot = payload.get("vae_drift_live_snapshot", {})
    rows = snapshot.get("rows", [])
    if rows:
        print("\nVAE-drift live snapshot:")
        for row in rows:
            last = row.get("last_eval") or {}
            best = row.get("best_eval") or {}
            print(
                f"  {row.get('experiment')}: "
                f"last_epoch={last.get('epoch', '-')} last_rho={last.get('spearman_rho', '-')} "
                f"best_epoch={best.get('epoch', '-')} best_rho={best.get('spearman_rho', '-')} "
                f"best_U={best.get('uniqueness', '-')} gate={best.get('gate', '-')}"
            )

    print("\nRunner status:")
    if not payload["runner_statuses"]:
        print("  no parallel runner status files")
    for status in payload["runner_statuses"]:
        devices = ",".join(status.get("devices", []))
        print(
            f"  {status['path']}: state={status['state']} selected={status['selected']} "
            f"runnable={status['runnable']} devices={devices}"
        )
        for item in status.get("running", []):
            print(
                f"    running gpu{item.get('device')}: {item.get('name')} "
                f"progress={item.get('progress')} eta={item.get('eta')} log={item.get('log')}"
            )
            if item.get("last_signal"):
                print(f"      last: {item.get('last_signal')[:140]}")
        for item in status.get("failures", []):
            print(f"    failed: {item}")

    print("\nReviewer faithful runner status:")
    if not payload["reviewer_runner_statuses"]:
        print("  no reviewer faithful status files")
    for status in payload["reviewer_runner_statuses"]:
        devices = ",".join(status.get("devices", []))
        print(
            f"  {status['path']}: state={status['state']} selected={status['selected']} "
            f"runnable={status['runnable']} devices={devices}"
        )
        for item in status.get("running", []):
            print(
                f"    running gpu{item.get('device')}: {item.get('name')} "
                f"progress={item.get('progress')} eta={item.get('eta')} log={item.get('log')}"
            )
            if item.get("last_signal"):
                print(f"      last: {item.get('last_signal')[:140]}")
        for item in status.get("failures", []):
            print(f"    failed: {item}")

    print("\nGPU status:")
    if not payload["gpus"]:
        print("  nvidia-smi unavailable")
    for gpu in payload["gpus"]:
        print(
            f"  gpu{gpu['index']}: util={gpu['utilization_gpu_pct']}% "
            f"mem={gpu['memory_used_mb']}/{gpu['memory_total_mb']} MiB {gpu['name']}"
        )
    disk = payload.get("disk") or {}
    if disk:
        print(
            "\nDisk status: "
            f"{disk.get('available')} available / {disk.get('size')} total "
            f"({disk.get('use_percent')} used) on {disk.get('mounted_on')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_table(payload)


if __name__ == "__main__":
    main()
