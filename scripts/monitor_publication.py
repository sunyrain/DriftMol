#!/usr/bin/env python3
"""Summarize publication experiment progress from status files and logs."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "outputs" / "publication"
LOG_DIR = PUB / "logs"

EPOCH_RE = re.compile(r"\[epoch\s+(?P<epoch>\d+)\]")
TRAIN_RE = re.compile(r"\[train\]\s+(?P<epochs>\d+)\s+epochs")
DURATION_RE = re.compile(r"\((?P<seconds>\d+)s\)")
STALE_LOG_SECONDS = 15 * 60


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


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
        elif line.startswith("[done]"):
            last_signal = "[done]"
        elif line.startswith("[final]"):
            last_signal = line.strip()
    return last_epoch, total_epochs, last_signal


def format_eta(last_epoch: int | None, total_epochs: int | None, signal: str, final: bool) -> str:
    if final:
        return "-"
    if last_epoch is None or total_epochs is None:
        return "-"
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


def format_age(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds}s"
    minutes, _ = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def log_age_seconds(log_path: Path) -> int | None:
    try:
        return max(0, int(time.time() - log_path.stat().st_mtime))
    except OSError:
        return None


def status_rows() -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for status_path in sorted(PUB.glob("runner_status*_*.json")) + sorted(PUB.glob("runner_status.json")):
        status = read_json(status_path)
        entry = status.get("entry", {})
        if status.get("state") == "completed" and not entry:
            continue
        name = entry.get("name", "-")
        out_dir = ROOT / entry.get("output_dir", "")
        log_path = ROOT / status.get("log", "") if status.get("log") else LOG_DIR / f"{name}.log"
        final_metrics = out_dir / "final_metrics.json"
        if final_metrics.exists() and status.get("state") == "running":
            try:
                if status_path.stat().st_mtime < final_metrics.stat().st_mtime:
                    continue
            except OSError:
                pass
        last_epoch, total_epochs, signal = parse_log(log_path)
        progress = "-"
        if last_epoch is not None and total_epochs is not None:
            progress = f"{last_epoch}/{total_epochs}"
        elif last_epoch is not None:
            progress = str(last_epoch)
        final_exists = final_metrics.exists()
        age_seconds = log_age_seconds(log_path)
        stale = (
            status.get("state") == "running"
            and not final_exists
            and age_seconds is not None
            and age_seconds > STALE_LOG_SECONDS
        )
        row = {
            "status_file": str(status_path.relative_to(ROOT)),
            "state": status.get("state", "-"),
            "name": name,
            "gpu": str(status.get("cuda_visible_devices", "-")),
            "progress": progress,
            "eta": format_eta(last_epoch, total_epochs, signal, final_exists),
            "age": format_age(age_seconds),
            "stale": "yes" if stale else "no",
            "final": "yes" if final_exists else "no",
            "signal": signal[:120],
            "updated_at": status.get("updated_at", ""),
        }
        key = (name, str(log_path))
        previous = by_key.get(key)
        if previous is None or row["updated_at"] > previous["updated_at"]:
            by_key[key] = row
    return sorted(by_key.values(), key=lambda r: (r["name"], r["status_file"]))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def watcher_rows() -> list[dict[str, str]]:
    rows = []
    for pid_path in sorted(PUB.glob("runner_followup_gpu*.pid")):
        gpu_match = re.search(r"gpu(?P<gpu>\d+)", pid_path.name)
        gpu = gpu_match.group("gpu") if gpu_match else "-"
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = -1
        status_path = PUB / f"runner_status_followup_gpu{gpu}.json"
        log_path = PUB / f"runner_followup_gpu{gpu}.log"
        status = read_json(status_path)
        entry = status.get("entry", {})
        state = status.get("state", "waiting" if pid_alive(pid) else "dead")
        name = entry.get("name", "waiting")
        rows.append({
            "gpu": gpu,
            "pid": str(pid),
            "alive": "yes" if pid_alive(pid) else "no",
            "state": state,
            "name": name,
            "status_exists": "yes" if status_path.exists() else "no",
            "log_exists": "yes" if log_path.exists() else "no",
            "status_file": str(status_path.relative_to(ROOT)),
            "pid_file": str(pid_path.relative_to(ROOT)),
        })
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    headers = ["state", "name", "gpu", "progress", "eta", "age", "stale", "final", "status_file"]
    widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}
    print(" ".join(h.ljust(widths[h]) for h in headers))
    print(" ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" ".join(row[h].ljust(widths[h]) for h in headers))
        if row["signal"]:
            print(f"  last: {row['signal']}")


def print_watcher_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No follow-up watcher pid files found.")
        return
    headers = ["gpu", "pid", "alive", "state", "name", "status_exists", "log_exists", "status_file"]
    widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}
    print(" ".join(h.ljust(widths[h]) for h in headers))
    print(" ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" ".join(row[h].ljust(widths[h]) for h in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON rows.")
    parser.add_argument("--watchers", action="store_true", help="Also report follow-up watcher PID liveness.")
    args = parser.parse_args()

    rows = status_rows()
    if args.json:
        payload = {"experiments": rows, "watchers": watcher_rows()} if args.watchers else rows
        print(json.dumps(payload, indent=2))
        return
    if not rows:
        print("No publication runner status files found.")
    else:
        print_table(rows)
    if args.watchers:
        print()
        print_watcher_table(watcher_rows())


if __name__ == "__main__":
    main()
