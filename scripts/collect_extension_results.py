#!/usr/bin/env python3
"""Collect extension-stage experiment results into review-ready artifacts."""
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


DEFAULT_MANIFEST = ROOT / "configs" / "publication_ext" / "manifest.json"
OUT_CSV = ROOT / "results" / "destructive_ablation.csv"
OUT_STATUS = ROOT / "results" / "destructive_ablation_status.json"
OUT_TEX = ROOT / "results" / "tables" / "tab_destructive_ablation.tex"
OUT_CHECKLIST = ROOT / "docs" / "EXTENSION_EXECUTION_CHECKLIST.md"

DISPLAY_NAMES = {
    "D_ATTR": "D-ATTR",
    "D_REPL": "D-REPL",
    "D_BROKEN_ATTR": "D-BROKEN-A",
    "D_BROKEN_REPL": "D-BROKEN-R",
    "D_YONLY": "D-YONLY",
    "D_NOCROSS": "D-NOCROSS",
    "D_NONORM": "D-NONORM",
}

CHANGE_LABELS = {
    "D_ATTR": "attraction only",
    "D_REPL": "repulsion only",
    "D_BROKEN_ATTR": "1.5x attraction",
    "D_BROKEN_REPL": "1.5x repulsion",
    "D_YONLY": "y-only normalization",
    "D_NOCROSS": "no cross-multiplication",
    "D_NONORM": "no normalization",
}


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


def collect_manifest_rows(
    manifest_path: Path,
    min_validity: float,
    min_uniqueness: float,
    status_root: Path,
) -> list[tuple[collect.ExperimentRow, dict[str, Any]]]:
    entries = collect.load_manifest_entries(manifest_path)
    pairs: list[tuple[collect.ExperimentRow, dict[str, Any]]] = []
    for entry in entries:
        output_dir = ROOT / entry["output_dir"]
        if entry.get("group") == "vae_sensitivity":
            row = collect.manifest_row(entry)
            final_metrics = output_dir / "final_metrics.json"
            if final_metrics.exists():
                row.status = "complete_pass"
                row.metrics = collect._read_json(final_metrics)
        elif output_dir.exists():
            row = collect.load_row(
                output_dir,
                min_validity=min_validity,
                min_uniqueness=min_uniqueness,
            )
        else:
            row = collect.manifest_row(entry)
        collect.apply_manifest_metadata(row, entry)
        pairs.append((row, entry))
    collect.apply_running_status([row for row, _ in pairs], status_root)
    return pairs


def parallel_running_names() -> set[str]:
    names: set[str] = set()
    status_root = ROOT / "outputs" / "publication_ext"
    for path in status_root.glob("parallel_runner_status*.json"):
        payload = collect._read_json(path)
        if payload.get("state") != "running":
            continue
        for item in payload.get("running", []):
            name = item.get("name")
            if name:
                names.add(str(name))
    return names


def row_payload(row: collect.ExperimentRow, entry: dict[str, Any]) -> dict[str, Any]:
    metrics = row.metrics or {}
    return {
        "status": row.status,
        "group": row.manifest_group,
        "experiment": row.experiment,
        "variant": row.variant,
        "display": DISPLAY_NAMES.get(row.variant, row.variant),
        "change": CHANGE_LABELS.get(row.variant, entry.get("purpose", "")),
        "purpose": entry.get("purpose", ""),
        "seed": row.seed,
        "alpha": row.alpha,
        "spearman_rho": metrics.get("spearman_rho"),
        "validity": metrics.get("validity"),
        "uniqueness": metrics.get("uniqueness"),
        "novelty": metrics.get("novelty"),
        "mae": metrics.get("mae"),
        "slope": metrics.get("slope"),
        "warning": row.warning,
        "output_dir": row.output_dir,
        "config": entry.get("config", ""),
        "command": entry.get("command", ""),
    }


def write_csv(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "group",
        "experiment",
        "variant",
        "display",
        "change",
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
                "minimum_completed_runs_for_table": 3,
                "minimum_completed_runs_reached": len(complete) >= 3,
                "status_counts": dict(sorted(status_counts.items())),
                "pending_or_incomplete_entries": pending,
            },
            indent=2,
        )
        + "\n"
    )


def write_tex(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/collect_extension_results.py",
        "\\begin{tabular}{l l c c c c c}",
        "\\toprule",
        "Ablation & Change & Status & $\\rho$ & U (\\%) & MAE & Slope \\\\",
        "\\midrule",
    ]
    for payload in payloads:
        lines.append(
            f"{tex_escape(payload['display'])} & {tex_escape(payload['change'])} & "
            f"{tex_escape(payload['status'])} & {fnum(payload.get('spearman_rho'))} & "
            f"{fpct(payload.get('uniqueness'))} & {fnum(payload.get('mae'))} & "
            f"{fnum(payload.get('slope'))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_checklist(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Extension Execution Checklist",
        "",
        "Updated: 2026-05-13 UTC",
        "",
        "This checklist tracks extension-stage experiments that are intentionally",
        "separate from the audited 8-page AAAI submission package.",
        "",
        "## Destructive Drift Ablations",
        "",
        "| Group | Experiment | Change | Status | Command |",
        "|---|---|---|---|---|",
    ]
    for payload in payloads:
        lines.append(
            f"| {payload['group']} | `{payload['experiment']}` | {payload['change']} | {payload['status']} | "
            f"`{payload['command']}` |"
        )
    lines += [
        "",
        "Completion gate: at least three destructive ablations should complete and",
        "act as interpretable negative controls before this evidence is promoted into",
        "the main paper.",
    ]
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

    pairs = collect_manifest_rows(
        manifest_path=manifest_path,
        min_validity=args.min_validity,
        min_uniqueness=args.min_uniqueness,
        status_root=status_root,
    )
    payloads = [row_payload(row, entry) for row, entry in pairs]
    payloads.sort(key=lambda item: item["experiment"])
    running_names = parallel_running_names()
    for payload in payloads:
        if payload["experiment"] in running_names and payload["status"] == "pending":
            payload["status"] = "running_or_incomplete"
    destructive_payloads = [payload for payload in payloads if payload["group"] == "destructive_drift"]

    write_csv(OUT_CSV, destructive_payloads)
    write_status(OUT_STATUS, manifest_path, destructive_payloads)
    write_tex(OUT_TEX, destructive_payloads)
    write_checklist(OUT_CHECKLIST, payloads)

    print(f"Collected {len(payloads)} extension experiments")
    print(f"Collected {len(destructive_payloads)} destructive experiments")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_STATUS.relative_to(ROOT)}")
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")
    print(f"Wrote {OUT_CHECKLIST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
