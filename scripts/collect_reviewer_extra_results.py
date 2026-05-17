#!/usr/bin/env python3
"""Collect reviewer-extra experiment results."""
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


DEFAULT_MANIFEST = ROOT / "configs" / "publication_ext" / "reviewer_extra_manifest.json"
OUT_CSV = ROOT / "results" / "reviewer_extra_results.csv"
OUT_STATUS = ROOT / "results" / "reviewer_extra_status.json"
OUT_TEX = ROOT / "results" / "tables" / "tab_reviewer_extra.tex"


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
    for status_path in status_root.glob("parallel_runner_status_reviewer_extra*.json"):
        payload = collect._read_json(status_path)
        if payload.get("state") != "running":
            continue
        for item in payload.get("running", []):
            name = item.get("name") if isinstance(item, dict) else None
            if name:
                names.add(str(name))
    return names


def load_payloads(
    manifest_path: Path,
    status_root: Path,
    min_validity: float,
    min_uniqueness: float,
) -> list[dict[str, Any]]:
    running = parallel_running_names(status_root)
    payloads: list[dict[str, Any]] = []
    for entry in collect.load_manifest_entries(manifest_path):
        output_dir = ROOT / entry["output_dir"]
        if output_dir.exists():
            row = collect.load_row(output_dir, min_validity=min_validity, min_uniqueness=min_uniqueness)
        else:
            row = collect.manifest_row(entry)
        collect.apply_manifest_metadata(row, entry)
        if row.status == "pending" and entry["name"] in running:
            row.status = "running_or_incomplete"
        metrics = row.metrics or {}
        payloads.append(
            {
                "status": row.status,
                "group": entry.get("group", ""),
                "experiment": entry["name"],
                "display": entry.get("display", entry["name"]),
                "target_property": entry.get("target_property", ""),
                "purpose": entry.get("purpose", ""),
                "seed": row.seed or collect.parse_seed_from_name(entry["name"]),
                "alpha": row.alpha,
                "spearman_rho": metrics.get("spearman_rho"),
                "validity": metrics.get("validity"),
                "uniqueness": metrics.get("uniqueness"),
                "novelty": metrics.get("novelty"),
                "mae": metrics.get("mae"),
                "slope": metrics.get("slope"),
                "success_0p10": metrics.get("success_0p10"),
                "warning": row.warning,
                "output_dir": entry.get("output_dir", row.output_dir),
                "config": entry.get("config", ""),
                "command": entry.get("command", ""),
            }
        )
    payloads.sort(key=lambda item: (str(item.get("group")), str(item.get("target_property")), item.get("seed") or 999))
    return payloads


def write_csv(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "group",
        "experiment",
        "display",
        "target_property",
        "purpose",
        "seed",
        "alpha",
        "spearman_rho",
        "validity",
        "uniqueness",
        "novelty",
        "mae",
        "slope",
        "success_0p10",
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
                "pending_or_incomplete_entries": pending,
            },
            indent=2,
        )
        + "\n"
    )


def write_tex(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/collect_reviewer_extra_results.py",
        "\\begin{tabular}{l l c c c c c}",
        "\\toprule",
        "Experiment & Target & Seed & $\\alpha$ & $\\rho$ & U (\\%) & MAE \\\\",
        "\\midrule",
    ]
    for payload in payloads:
        lines.append(
            f"{tex_escape(payload.get('display'))} & "
            f"{tex_escape(payload.get('target_property', '---'))} & "
            f"{payload.get('seed', '---')} & {tex_escape(payload.get('alpha') or '---')} & "
            f"{fnum(payload.get('spearman_rho'))} & {fpct(payload.get('uniqueness'))} & "
            f"{fnum(payload.get('mae'))} \\\\"
        )
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

    payloads = load_payloads(
        manifest_path=manifest_path,
        status_root=status_root,
        min_validity=args.min_validity,
        min_uniqueness=args.min_uniqueness,
    )
    write_csv(OUT_CSV, payloads)
    write_status(OUT_STATUS, manifest_path, payloads)
    write_tex(OUT_TEX, payloads)
    print(f"Collected {len(payloads)} reviewer-extra entries")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_STATUS.relative_to(ROOT)}")
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
