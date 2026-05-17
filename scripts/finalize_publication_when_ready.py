#!/usr/bin/env python3
"""Finalize publication artifacts once all queued experiments are complete.

This helper is intentionally conservative: it refreshes result artifacts while
training continues, waits until the publication status reports no pending
experiments, then waits for an actually quiet GPU before running the inference
throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results" / "publication_status.json"
BENCHMARK = ROOT / "results" / "inference_benchmark.json"


@dataclass(frozen=True)
class GpuStat:
    index: int
    util_pct: int
    memory_mb: int


def log(message: str) -> None:
    print(f"[finalize] {message}", flush=True)


def run_cmd(args: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    log("$ " + " ".join(args))
    if dry_run:
        return
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def refresh_commands() -> list[list[str]]:
    return [
        [sys.executable, "scripts/collect_results.py"],
        [sys.executable, "scripts/export_latex_tables.py"],
        [sys.executable, "scripts/plot_main_figure.py"],
        [sys.executable, "scripts/plot_result_figures.py"],
    ]


def refresh_outputs(*, dry_run: bool = False, audit: bool = False) -> int:
    for cmd in refresh_commands():
        run_cmd(cmd, dry_run=dry_run)
    if audit:
        cmd = [sys.executable, "scripts/audit_publication_completion.py", "--run-tests"]
        log("$ " + " ".join(cmd))
        if dry_run:
            return 0
        proc = subprocess.run(cmd, cwd=ROOT)
        return proc.returncode
    return 0


def pending_count(status: dict) -> int | None:
    value = status.get("pending_or_incomplete")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def qed_seed_summary(status: dict) -> str:
    rows = sorted(status.get("qed_3seed", []), key=lambda row: row.get("variant", ""))
    if not rows:
        return "none"
    return ", ".join(f"{row.get('variant')}={row.get('n', 0)}/3" for row in rows)


def experiments_complete(status: dict) -> bool:
    return pending_count(status) == 0


def parse_gpu_stats(text: str) -> list[GpuStat]:
    stats: list[GpuStat] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            stats.append(GpuStat(int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return stats


def query_gpu_stats() -> list[GpuStat]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        log(f"nvidia-smi unavailable: {proc.stderr.strip() or proc.returncode}")
        return []
    return parse_gpu_stats(proc.stdout)


def choose_idle_gpu(stats: list[GpuStat], *, max_util_pct: int, max_memory_mb: int) -> GpuStat | None:
    candidates = [
        stat for stat in stats
        if stat.util_pct <= max_util_pct and stat.memory_mb <= max_memory_mb
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda stat: (stat.memory_mb, stat.util_pct, stat.index))[0]


def wait_for_experiments(args: argparse.Namespace) -> bool:
    start = time.monotonic()
    while True:
        refresh_outputs(dry_run=args.dry_run, audit=False)
        status = load_json(STATUS)
        pending = pending_count(status)
        log(f"pending_or_incomplete={pending}; qed seeds: {qed_seed_summary(status)}")
        if experiments_complete(status):
            return True
        if args.max_wait_seconds and time.monotonic() - start >= args.max_wait_seconds:
            log("max wait reached before experiments completed")
            return False
        time.sleep(args.poll_seconds)


def wait_for_idle_gpu(args: argparse.Namespace) -> GpuStat | None:
    start = time.monotonic()
    while True:
        stats = query_gpu_stats()
        idle = choose_idle_gpu(
            stats,
            max_util_pct=args.max_gpu_util,
            max_memory_mb=args.max_gpu_memory,
        )
        summary = ", ".join(
            f"{s.index}:util={s.util_pct}%,mem={s.memory_mb}MB" for s in stats
        ) or "no gpu stats"
        if idle is not None:
            log(f"selected idle GPU {idle.index}; {summary}")
            return idle
        log(f"waiting for quiet GPU; {summary}")
        if args.max_wait_seconds and time.monotonic() - start >= args.max_wait_seconds:
            log("max wait reached before a quiet GPU was available")
            return None
        time.sleep(args.gpu_poll_seconds)


def run_benchmark(args: argparse.Namespace, gpu: GpuStat) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    cmd = [
        sys.executable,
        "scripts/benchmark_inference.py",
        "--device",
        "cuda",
        "--output",
        str(BENCHMARK.relative_to(ROOT)),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--warmup-batches",
        str(args.warmup_batches),
    ]
    run_cmd(cmd, env=env, dry_run=args.dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--gpu-poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=0, help="0 means wait indefinitely.")
    parser.add_argument("--max-gpu-util", type=int, default=10)
    parser.add_argument("--max-gpu-memory", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--force-benchmark", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not wait_for_experiments(args):
        return 1

    if args.skip_benchmark:
        log("benchmark skipped by flag")
    elif BENCHMARK.exists() and not args.force_benchmark:
        log(f"benchmark already exists: {BENCHMARK.relative_to(ROOT)}")
    else:
        gpu = wait_for_idle_gpu(args)
        if gpu is None:
            return 1
        run_benchmark(args, gpu)

    if BENCHMARK.exists() or args.dry_run:
        run_cmd([sys.executable, "scripts/update_manuscript_benchmark.py"], dry_run=args.dry_run)

    return refresh_outputs(dry_run=args.dry_run, audit=True)


if __name__ == "__main__":
    raise SystemExit(main())
