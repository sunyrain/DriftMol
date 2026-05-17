#!/usr/bin/env python3
"""Run the graph-route chain after the graph VAE recovery process finishes."""
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
ARCHIVE = ROOT / "archive" / "graph_vae_line"
DEFAULT_LOG_DIR = ROOT / "outputs" / "publication_ext" / "graph_stress_logs"
DEFAULT_STATUS = ROOT / "outputs" / "publication_ext" / "graph_route_chain_status.json"
DEFAULT_VAE_PID = DEFAULT_LOG_DIR / "graph_recover_vae_v3_valence.pid"
DEFAULT_VAE_LOG = DEFAULT_LOG_DIR / "graph_recover_vae_v3_valence.log"
MANIFEST = ROOT / "configs" / "publication_ext" / "graph_stress_manifest.json"


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["updated_at"] = timestamp()
    path.write_text(json.dumps(out, indent=2) + "\n")


def pid_state(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return "missing"
    text = stat.read_text(errors="replace")
    parts = text.split()
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


def process_alive(pid: int) -> bool:
    state = pid_state(pid)
    return state not in {"missing", "Z", "X"}


def read_pid(path: Path) -> int:
    return int(path.read_text().strip())


def wait_for_vae(
    pid_file: Path,
    vae_log: Path,
    status_file: Path,
    poll_seconds: float,
) -> None:
    if not pid_file.exists():
        raise FileNotFoundError(pid_file)
    pid = read_pid(pid_file)
    while True:
        current_pid = read_pid(pid_file)
        if current_pid != pid:
            pid = current_pid
        if not process_alive(pid):
            break
        write_status(
            status_file,
            {
                "state": "waiting_for_graph_vae",
                "vae_pid": pid,
                "vae_pid_state": pid_state(pid),
                "vae_log": rel(vae_log),
            },
        )
        time.sleep(poll_seconds)

    log_text = vae_log.read_text(errors="replace") if vae_log.exists() else ""
    final_idx = log_text.rfind("[final] reconstruction + generation metrics")
    error_idx = max(log_text.rfind("Traceback"), log_text.rfind("OutOfMemoryError"))
    if final_idx < 0:
        raise RuntimeError(f"graph VAE process ended before final evaluation; inspect {rel(vae_log)}")
    if error_idx > final_idx:
        raise RuntimeError(f"graph VAE process ended with an error after final marker; inspect {rel(vae_log)}")


def run_step(
    name: str,
    command: str,
    log_dir: Path,
    status_file: Path,
    env: dict[str, str],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    write_status(
        status_file,
        {
            "state": "running_step",
            "step": name,
            "log": rel(log_path),
            "command": command,
        },
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== graph chain start {timestamp()} step={name} =====\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=ROOT,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log.write(f"\n===== graph chain end {timestamp()} step={name} rc={proc.returncode} =====\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with rc={proc.returncode}; inspect {rel(log_path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-pid-file", type=Path, default=DEFAULT_VAE_PID)
    parser.add_argument("--vae-log", type=Path, default=DEFAULT_VAE_LOG)
    parser.add_argument("--poll-seconds", type=float, default=1800.0)
    parser.add_argument("--device", default="2")
    parser.add_argument("--generator-devices", default="2,3")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()

    if args.pid_file:
        pid_path = args.pid_file if args.pid_file.is_absolute() else ROOT / args.pid_file
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(f"{os.getpid()}\n")

    vae_pid_file = args.vae_pid_file if args.vae_pid_file.is_absolute() else ROOT / args.vae_pid_file
    vae_log = args.vae_log if args.vae_log.is_absolute() else ROOT / args.vae_log
    log_dir = args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir
    status_file = args.status_file if args.status_file.is_absolute() else ROOT / args.status_file

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"

    try:
        wait_for_vae(vae_pid_file, vae_log, status_file, args.poll_seconds)

        env_one = dict(base_env)
        env_one["CUDA_VISIBLE_DEVICES"] = args.device
        env_one["PYTHONPATH"] = "."
        run_step(
            "graph_rebuild_latent_cache_v3",
            "cd archive/graph_vae_line && PYTHONPATH=. python scripts/build_latent_cache.py "
            "--vae_ckpt outputs/vae_v3_valence/best.pt --output data/cache/qm9_latent_cache_v3.pt",
            log_dir,
            status_file,
            env_one,
        )
        run_step(
            "graph_recover_latent_mae_v3",
            "cd archive/graph_vae_line && PYTHONPATH=. python -m src.train.train_latent_mae "
            "configs/publication/latent_mae_v3_recover.yaml",
            log_dir,
            status_file,
            env_one,
        )

        run_step(
            "graph_fresh_generators",
            f"{sys.executable} scripts/run_manifest_parallel.py --manifest {rel(MANIFEST)} "
            "--name graph_fresh_qed_e36 --name graph_fresh_logp_e40 --name graph_qed_destructive_no_drift "
            f"--devices {args.generator_devices} --poll-seconds 30 "
            "--status-file outputs/publication_ext/parallel_runner_status_graph_stress.json "
            "--log-dir outputs/publication_ext/graph_stress_logs",
            log_dir,
            status_file,
            base_env,
        )
        run_step(
            "graph_decode_and_compare",
            f"{sys.executable} scripts/run_manifest_parallel.py --manifest {rel(MANIFEST)} "
            "--name graph_raw_vs_repaired_decode --name graph_selfies_fair_comparison "
            f"--devices {args.device} --poll-seconds 30 "
            "--status-file outputs/publication_ext/parallel_runner_status_graph_postprocess.json "
            "--log-dir outputs/publication_ext/graph_stress_logs",
            log_dir,
            status_file,
            base_env,
        )
        run_step(
            "graph_launchability_audit",
            f"{sys.executable} scripts/audit_graph_archive_launchability.py",
            log_dir,
            status_file,
            base_env,
        )
    except Exception as exc:
        write_status(status_file, {"state": "failed", "error": str(exc)})
        print(f"[graph-chain] failed: {exc}", file=sys.stderr)
        return 1

    write_status(status_file, {"state": "completed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
