#!/usr/bin/env python3
"""Collect reviewer-facing generalization experiment results."""
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


DEFAULT_MANIFEST = ROOT / "configs" / "publication_ext" / "generalization_manifest.json"
OUT_CSV = ROOT / "results" / "generalization_results.csv"
OUT_STATUS = ROOT / "results" / "generalization_status.json"
OUT_TEX = ROOT / "results" / "tables" / "tab_generalization.tex"


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


def parallel_running_names(status_root: Path) -> set[str]:
    names: set[str] = set()
    for status_path in status_root.glob("parallel_runner_status_generalization*.json"):
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


def payload_for(row: collect.ExperimentRow, entry: dict[str, Any]) -> dict[str, Any]:
    metrics = row.metrics or {}
    target = entry.get("target_property", "")
    seed = row.seed if row.seed is not None else collect.parse_seed_from_name(entry["name"])
    payload = {
        "status": row.status,
        "group": entry.get("group", ""),
        "experiment": entry["name"],
        "target_property": target,
        "purpose": entry.get("purpose", ""),
        "seed": seed,
        "alpha": row.alpha,
        "warning": row.warning,
        "output_dir": entry.get("output_dir", row.output_dir),
        "config": entry.get("config", ""),
        "command": entry.get("command", ""),
    }
    for key in (
        "spearman_rho",
        "avg_spearman_rho",
        "validity",
        "uniqueness",
        "avg_uniqueness",
        "min_uniqueness",
        "novelty",
        "mae",
        "avg_mae",
        "slope",
        "avg_slope",
        "qed_rho",
        "sa_score_rho",
        "logp_rho",
        "molwt_rho",
        "qed_uniqueness",
        "sa_score_uniqueness",
        "logp_uniqueness",
        "molwt_uniqueness",
    ):
        payload[key] = metrics.get(key)
    return payload


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def aggregate(payloads: list[dict[str, Any]], group: str, key: str) -> dict[str, Any]:
    complete = [
        p for p in payloads
        if p.get("group") == group and str(p["status"]).startswith("complete")
    ]
    values = [float(p[key]) for p in complete if p.get(key) is not None]
    mean, std = mean_std(values)
    return {
        "n": len(complete),
        "seeds": ",".join(str(p["seed"]) for p in complete if p.get("seed") is not None),
        f"{key}_mean": mean,
        f"{key}_std": std,
    }


def write_csv(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "group",
        "experiment",
        "target_property",
        "purpose",
        "seed",
        "alpha",
        "spearman_rho",
        "avg_spearman_rho",
        "validity",
        "uniqueness",
        "avg_uniqueness",
        "min_uniqueness",
        "novelty",
        "mae",
        "avg_mae",
        "slope",
        "avg_slope",
        "qed_rho",
        "sa_score_rho",
        "logp_rho",
        "molwt_rho",
        "qed_uniqueness",
        "sa_score_uniqueness",
        "logp_uniqueness",
        "molwt_uniqueness",
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
    group_counts: dict[str, dict[str, int]] = {}
    for payload in payloads:
        status = str(payload["status"])
        group = str(payload["group"])
        status_counts[status] = status_counts.get(status, 0) + 1
        group_counts.setdefault(group, {"total": 0, "complete": 0, "pending_or_incomplete": 0})
        group_counts[group]["total"] += 1
        if status.startswith("complete"):
            group_counts[group]["complete"] += 1
        else:
            group_counts[group]["pending_or_incomplete"] += 1
    path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path.relative_to(ROOT)),
                "num_experiments": len(payloads),
                "complete": len(complete),
                "pending_or_incomplete": len(pending),
                "status_counts": dict(sorted(status_counts.items())),
                "groups": group_counts,
                "multi4_avg_rho": aggregate(payloads, "multi4_seed_stability", "avg_spearman_rho"),
                "single_property_rho": {
                    p["target_property"]: p.get("spearman_rho")
                    for p in complete
                    if p.get("group") == "single_property_generalization"
                },
                "pending_or_incomplete_entries": pending,
            },
            indent=2,
        )
        + "\n"
    )


def write_tex(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    singles = [p for p in payloads if p.get("group") == "single_property_generalization"]
    multi = [p for p in payloads if p.get("group") == "multi4_seed_stability"]
    lines = [
        "% Generated by scripts/collect_generalization_results.py",
        "\\begin{tabular}{l c c c c c}",
        "\\toprule",
        "Target & Seed & $\\alpha$ & $\\rho$ & U (\\%) & MAE \\\\",
        "\\midrule",
    ]
    for payload in sorted(singles, key=lambda item: str(item.get("target_property"))):
        lines.append(
            f"{tex_escape(payload.get('target_property', '---'))} & "
            f"{payload.get('seed', '---')} & {tex_escape(payload.get('alpha') or '---')} & "
            f"{fnum(payload.get('spearman_rho'))} & {fpct(payload.get('uniqueness'))} & "
            f"{fnum(payload.get('mae'))} \\\\"
        )
    lines += [
        "\\midrule",
        "\\multicolumn{6}{l}{Four-property no-binning stability} \\\\",
        "Seed & $\\alpha$ & Avg $\\rho$ & QED / SA / LogP / MW $\\rho$ & Lowest U (\\%) & Status \\\\",
        "\\midrule",
    ]
    for payload in sorted(multi, key=lambda item: item.get("seed") or 999):
        prop_rhos = " / ".join(
            fnum(payload.get(key))
            for key in ("qed_rho", "sa_score_rho", "logp_rho", "molwt_rho")
        )
        lines.append(
            f"{payload.get('seed', '---')} & {tex_escape(payload.get('alpha') or '---')} & "
            f"{fnum(payload.get('avg_spearman_rho'))} & {prop_rhos} & "
            f"{fpct(payload.get('min_uniqueness'))} & {tex_escape(payload.get('status'))} \\\\"
        )
    summary = aggregate(payloads, "multi4_seed_stability", "avg_spearman_rho")
    if summary["n"]:
        lines += [
            "\\midrule",
            f"mean ({summary['n']} seeds) & --- & "
            f"{fnum(summary.get('avg_spearman_rho_mean'))}$\\pm${fnum(summary.get('avg_spearman_rho_std'))} & "
            "--- & --- & --- \\\\",
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
    payloads.sort(key=lambda item: (str(item.get("group")), str(item.get("target_property")), item.get("seed") or 999))

    write_csv(OUT_CSV, payloads)
    write_status(OUT_STATUS, manifest_path, payloads)
    write_tex(OUT_TEX, payloads)

    print(f"Collected {len(payloads)} generalization entries")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_STATUS.relative_to(ROOT)}")
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
