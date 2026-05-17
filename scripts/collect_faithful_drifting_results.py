#!/usr/bin/env python3
"""Collect reviewer-faithful Drifting reproduction results."""
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
from scripts import render_faithful_supplement


DEFAULT_MANIFEST = ROOT / "configs" / "reviewer_faithful" / "manifest.json"
OUT_CSV = ROOT / "results" / "faithful_drifting.csv"
OUT_STATUS = ROOT / "results" / "faithful_drifting_status.json"
OUT_CORE_TEX = ROOT / "results" / "tables" / "tab_faithful_drifting_core.tex"
OUT_ALLOC_TEX = ROOT / "results" / "tables" / "tab_faithful_drifting_allocation.tex"
OUT_CHECKLIST = ROOT / "docs" / "DRIFTING_FAITHFULNESS_EXECUTION_CHECKLIST.md"


DISPLAY_NAMES = {
    "rf_FD_STRICT_PLAIN_PHI_QED_s42": "Strict plain-$\\phi$",
    "rf_FD_STRICT_PROP_PHI_QED_s42": "Property-aware $\\phi$",
    "rf_FD_STRICT_RANDOM_PHI_QED_s42": "Random $\\phi$",
    "rf_FD_STRICT_ZSPACE_QED_s42": "z-space drift",
    "rf_FD_ALLOC_POS01_QED_s42": "$N_{pos}=1$",
    "rf_FD_ALLOC_POS16_QED_s42": "$N_{pos}=16$",
    "rf_FD_ALLOC_POS32_QED_s42": "$N_{pos}=32$",
    "rf_FD_ALLOC_POS64_QED_s42": "$N_{pos}=64$",
    "rf_FD_ALLOC_NEG16_QED_s42": "$N_{neg}=16$",
    "rf_FD_ALLOC_NEG32_QED_s42": "$N_{neg}=32$",
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
    text = str(value)
    # Preserve simple math labels used in DISPLAY_NAMES.
    if "$" in text or "\\" in text:
        return text
    return text.replace("_", "\\_")


def load_pairs(
    manifest_path: Path,
    status_root: Path,
    min_validity: float,
    min_uniqueness: float,
) -> list[tuple[collect.ExperimentRow, dict[str, Any]]]:
    entries = collect.load_manifest_entries(manifest_path)
    pairs: list[tuple[collect.ExperimentRow, dict[str, Any]]] = []
    for entry in entries:
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
    collect.apply_running_status([row for row, _ in pairs], status_root)
    return pairs


def payload_for(row: collect.ExperimentRow, entry: dict[str, Any]) -> dict[str, Any]:
    metrics = row.metrics or {}
    name = entry["name"]
    return {
        "status": row.status,
        "group": entry.get("group", ""),
        "experiment": name,
        "display": DISPLAY_NAMES.get(name, name),
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
        "config": entry.get("config", ""),
        "output_dir": entry.get("output_dir", ""),
        "command": entry.get("command", ""),
    }


def running_names(status_root: Path) -> set[str]:
    names: set[str] = set()
    for status_path in status_root.glob("*status.json"):
        try:
            payload = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("state") != "running":
            continue
        for item in payload.get("running", []):
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
    return names


def apply_running_names(payloads: list[dict[str, Any]], status_root: Path) -> None:
    names = running_names(status_root)
    if not names:
        return
    for payload in payloads:
        if payload["status"] == "pending" and payload["experiment"] in names:
            payload["status"] = "running_or_incomplete"


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
        "config",
        "output_dir",
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
    groups: dict[str, dict[str, int]] = {}
    for payload in payloads:
        group = str(payload["group"])
        groups.setdefault(group, {"total": 0, "complete": 0, "pending": 0})
        groups[group]["total"] += 1
        if str(payload["status"]).startswith("complete"):
            groups[group]["complete"] += 1
        else:
            groups[group]["pending"] += 1
    path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path.relative_to(ROOT)),
                "num_experiments": len(payloads),
                "complete": len(complete),
                "pending_or_incomplete": len(pending),
                "faithful_core_complete": groups.get("faithful_core", {}).get("pending", 0) == 0
                and groups.get("faithful_core", {}).get("total", 0) > 0,
                "groups": groups,
                "pending_or_incomplete_entries": pending,
            },
            indent=2,
        )
        + "\n"
    )


def write_tex(path: Path, payloads: list[dict[str, Any]], caption_comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"% Generated by scripts/collect_faithful_drifting_results.py: {caption_comment}",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        "Run & Status & $\\alpha$ & $\\rho$ & U (\\%) & MAE & Slope \\\\",
        "\\midrule",
    ]
    for payload in payloads:
        lines.append(
            f"{tex_escape(payload['display'])} & {tex_escape(payload['status'])} & "
            f"{tex_escape(payload.get('alpha') or '---')} & {fnum(payload.get('spearman_rho'))} & "
            f"{fpct(payload.get('uniqueness'))} & {fnum(payload.get('mae'))} & "
            f"{fnum(payload.get('slope'))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_checklist(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core_total = sum(1 for payload in payloads if payload["group"] == "faithful_core")
    core_done = sum(
        1
        for payload in payloads
        if payload["group"] == "faithful_core"
        and str(payload["status"]).startswith("complete")
    )
    alloc_total = sum(1 for payload in payloads if payload["group"] == "faithful_allocation")
    alloc_done = sum(
        1
        for payload in payloads
        if payload["group"] == "faithful_allocation"
        and str(payload["status"]).startswith("complete")
    )
    core_complete = core_total > 0 and core_done == core_total
    alloc_complete = alloc_total > 0 and alloc_done == alloc_total
    if core_complete and alloc_complete:
        gate_lines = [
            "- The faithful-reproduction package is complete for the strict core",
            "  and allocation sweep; promote it only as conservative supplemental",
            "  evidence because target tracking remains modest.",
            "- Allocation sweeps are secondary mechanism evidence: they improve",
            "  diversity more reliably than QED target tracking.",
        ]
        completion_lines = [
            "Completion gate: the strict core and allocation rows are complete.",
            "The faithful-reproduction claim may be described as completed",
            "supplemental evidence, with the conservative interpretation stated",
            "above.",
        ]
    else:
        gate_lines = [
            "- Do not promote a faithful-reproduction claim until every",
            "  `faithful_core` row is complete and the faithfulness audits close.",
            "- Allocation sweeps remain secondary evidence; run or interpret them after",
            "  the strict core evidence is informative.",
        ]
        completion_lines = [
            "Completion gate: all `faithful_core` runs must complete before the",
            "faithful-reproduction claim is promoted from planned supplemental evidence",
            "to completed supplemental evidence.",
        ]
    lines = [
        "# Drifting Faithfulness Execution Checklist",
        "",
        "This checklist tracks reviewer-facing faithful Drifting reproduction runs.",
        "They are supplemental and separate from the stable 8-page AAAI draft.",
        "",
        "## Current Gate Summary",
        "",
        f"- Strict faithful core: {core_done}/{core_total} complete.",
        f"- Allocation sweeps: {alloc_done}/{alloc_total} complete.",
        *gate_lines,
        "",
        "## Run Matrix",
        "",
        "| Group | Experiment | Status | Command |",
        "|---|---|---|---|",
    ]
    for payload in payloads:
        lines.append(
            f"| {payload['group']} | `{payload['experiment']}` | {payload['status']} | "
            f"`{payload['command']}` |"
        )
    lines += [
        "",
        "## Automation And Audit Commands",
        "",
        "Deferred core launcher:",
        "",
        "```bash",
        "python scripts/defer_faithful_core_after_destructive.py \\",
        "  --watch-status outputs/publication_ext/parallel_runner_status.json \\",
        "  --faithful-status outputs/reviewer_faithful/core_status.json \\",
        "  --devices 0,2,3 \\",
        "  --poll-seconds 60 \\",
        "  --log-dir outputs/reviewer_faithful/logs \\",
        "  --pid-file outputs/reviewer_faithful/deferred_faithful_core_launcher.pid",
        "```",
        "",
        "Completion audit:",
        "",
        "```bash",
        "python scripts/collect_faithful_drifting_results.py",
        "python scripts/audit_drifting_faithfulness.py",
        "python scripts/audit_reviewer_experiment_readiness.py",
        "```",
        "",
        *completion_lines,
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--status-root", default="outputs/reviewer_faithful")
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
        manifest_path,
        status_root=status_root,
        min_validity=args.min_validity,
        min_uniqueness=args.min_uniqueness,
    )
    payloads = [payload_for(row, entry) for row, entry in pairs]
    apply_running_names(payloads, status_root)
    payloads.sort(key=lambda p: (p["group"], p["experiment"]))

    write_csv(OUT_CSV, payloads)
    write_status(OUT_STATUS, manifest_path, payloads)
    write_tex(
        OUT_CORE_TEX,
        [p for p in payloads if p["group"] == "faithful_core"],
        "strict core faithfulness runs",
    )
    write_tex(
        OUT_ALLOC_TEX,
        [p for p in payloads if p["group"] == "faithful_allocation"],
        "positive/negative allocation runs",
    )
    write_checklist(OUT_CHECKLIST, payloads)
    render_faithful_supplement.render()

    print(f"Collected {len(payloads)} faithful Drifting experiments")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_STATUS.relative_to(ROOT)}")
    print(f"Wrote {OUT_CORE_TEX.relative_to(ROOT)}")
    print(f"Wrote {OUT_ALLOC_TEX.relative_to(ROOT)}")
    print(f"Wrote {OUT_CHECKLIST.relative_to(ROOT)}")
    print(f"Wrote {render_faithful_supplement.OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
