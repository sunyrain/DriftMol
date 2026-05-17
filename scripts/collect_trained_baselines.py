#!/usr/bin/env python3
"""Collect reviewer-facing trained baseline results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_results as collect


DEFAULT_MANIFEST = ROOT / "configs" / "publication_ext" / "baseline_manifest.json"
OUT_CSV = ROOT / "results" / "trained_baseline_qed.csv"
OUT_STATUS = ROOT / "results" / "trained_baseline_status.json"
OUT_TEX = ROOT / "results" / "tables" / "tab_trained_baseline_qed.tex"


def fnum(value: Any, digits: int = 3) -> str:
    if value in (None, "", "-"):
        return "---"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val):
        return "---"
    return f"{val:.{digits}f}"


def fpct(value: Any, digits: int = 1) -> str:
    if value in (None, "", "-"):
        return "---"
    try:
        val = 100.0 * float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val):
        return "---"
    return f"{val:.{digits}f}"


def tex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_")


def load_pairs(
    manifest_path: Path,
    status_root: Path,
    min_validity: float,
    min_uniqueness: float,
) -> list[tuple[collect.ExperimentRow, dict[str, Any]]]:
    pairs: list[tuple[collect.ExperimentRow, dict[str, Any]]] = []
    for entry in collect.load_manifest_entries(manifest_path):
        output_dir = ROOT / entry["output_dir"]
        if output_dir.exists():
            row = collect.load_row(
                output_dir,
                min_validity=min_validity,
                min_uniqueness=min_uniqueness,
            )
        else:
            row = collect.manifest_row(entry)
        collect.apply_manifest_metadata(row, entry)
        pairs.append((row, entry))
    apply_parallel_running([row for row, _ in pairs], status_root)
    return pairs


def parallel_running_names(status_root: Path) -> set[str]:
    names: set[str] = set()
    for status_path in status_root.glob("parallel_runner_status_baseline*.json"):
        payload = collect._read_json(status_path)
        if payload.get("state") != "running":
            continue
        for item in payload.get("running", []):
            name = item.get("name") if isinstance(item, dict) else None
            if name:
                names.add(str(name))
    return names


def apply_parallel_running(rows: list[collect.ExperimentRow], status_root: Path) -> None:
    names = parallel_running_names(status_root)
    if not names:
        return
    for row in rows:
        if row.status == "pending" and row.experiment in names:
            row.status = "running_or_incomplete"


def payload_for(row: collect.ExperimentRow, entry: dict[str, Any]) -> dict[str, Any]:
    metrics = row.metrics or {}
    seed = row.seed if row.seed is not None else collect.parse_seed_from_name(entry["name"])
    return {
        "status": row.status,
        "group": entry.get("group", ""),
        "experiment": entry["name"],
        "display": "Linear property guidance",
        "purpose": entry.get("purpose", ""),
        "seed": seed,
        "alpha": row.alpha,
        "spearman_rho": metrics.get("spearman_rho"),
        "validity": metrics.get("validity"),
        "uniqueness": metrics.get("uniqueness"),
        "novelty": metrics.get("novelty"),
        "mae": metrics.get("mae"),
        "slope": metrics.get("slope"),
        "warning": row.warning,
        "output_dir": entry.get("output_dir", row.output_dir),
        "config": entry.get("config", ""),
        "command": entry.get("command", ""),
    }


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def aggregate(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [p for p in payloads if str(p["status"]).startswith("complete")]
    def vals(key: str) -> list[float]:
        return [float(p[key]) for p in complete if p.get(key) is not None]

    rho_mean, rho_std = mean_std(vals("spearman_rho"))
    u_mean, u_std = mean_std(vals("uniqueness"))
    mae_mean, mae_std = mean_std(vals("mae"))
    slope_mean, slope_std = mean_std(vals("slope"))
    return {
        "n": len(complete),
        "seeds": ",".join(str(p["seed"]) for p in complete if p.get("seed") is not None),
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "uniqueness_mean": u_mean,
        "uniqueness_std": u_std,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "slope_mean": slope_mean,
        "slope_std": slope_std,
    }


def write_csv(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "group",
        "experiment",
        "display",
        "purpose",
        "seed",
        "alpha",
        "spearman_rho",
        "validity",
        "uniqueness",
        "novelty",
        "mae",
        "slope",
        "warning",
        "output_dir",
        "config",
        "command",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(payloads)


def write_status(path: Path, manifest_path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = [p for p in payloads if str(p["status"]).startswith("complete")]
    pending = [p for p in payloads if not str(p["status"]).startswith("complete")]
    status_counts: dict[str, int] = {}
    for payload in payloads:
        status = str(payload["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path.relative_to(ROOT)),
                "num_experiments": len(payloads),
                "complete": len(complete),
                "pending_or_incomplete": len(pending),
                "three_seed_complete": len(complete) >= 3,
                "status_counts": dict(sorted(status_counts.items())),
                "aggregate": aggregate(payloads),
                "pending_or_incomplete_entries": pending,
            },
            indent=2,
        )
        + "\n"
    )


def write_tex(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(payloads, key=lambda item: item.get("seed") or 999)
    lines = [
        "% Generated by scripts/collect_trained_baselines.py",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        "Seed & Status & $\\alpha$ & $\\rho$ & U (\\%) & MAE & Slope \\\\",
        "\\midrule",
    ]
    for payload in rows:
        lines.append(
            f"{payload.get('seed', '---')} & {tex_escape(payload['status'])} & "
            f"{tex_escape(payload.get('alpha') or '---')} & {fnum(payload.get('spearman_rho'))} & "
            f"{fpct(payload.get('uniqueness'))} & {fnum(payload.get('mae'))} & "
            f"{fnum(payload.get('slope'))} \\\\"
        )
    summary = aggregate(payloads)
    if summary["n"]:
        lines += [
            "\\midrule",
            f"mean ({summary['n']} seeds) & --- & --- & "
            f"{fnum(summary.get('rho_mean'))}$\\pm${fnum(summary.get('rho_std'))} & "
            f"{fpct(summary.get('uniqueness_mean'))}$\\pm${fpct(summary.get('uniqueness_std'))} & "
            f"{fnum(summary.get('mae_mean'))}$\\pm${fnum(summary.get('mae_std'))} & "
            f"{fnum(summary.get('slope_mean'))}$\\pm${fnum(summary.get('slope_std'))} \\\\",
        ]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--status-root", default="outputs/publication_ext")
    parser.add_argument("--min-validity", type=float, default=0.95)
    parser.add_argument("--min-uniqueness", type=float, default=0.10)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    status_root = Path(args.status_root)
    if not status_root.is_absolute():
        status_root = ROOT / status_root

    pairs = load_pairs(
        manifest_path=manifest_path,
        status_root=status_root,
        min_validity=args.min_validity,
        min_uniqueness=args.min_uniqueness,
    )
    payloads = [payload_for(row, entry) for row, entry in pairs]
    payloads.sort(key=lambda item: item.get("seed") or 999)

    write_csv(OUT_CSV, payloads)
    write_status(OUT_STATUS, manifest_path, payloads)
    write_tex(OUT_TEX, payloads)

    print(f"Collected {len(payloads)} trained baseline entries")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_STATUS.relative_to(ROOT)}")
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
