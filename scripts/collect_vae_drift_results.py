#!/usr/bin/env python3
"""Collect downstream drifting results for alternative SELFIES VAEs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "publication_ext" / "vae_drift_manifest.json"
OUT_CSV = ROOT / "results" / "vae_drift_downstream.csv"
OUT_STATUS = ROOT / "results" / "vae_drift_downstream_status.json"
OUT_TEX = ROOT / "results" / "tables" / "tab_vae_drift_downstream.tex"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "---"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "---"
    return f"{x:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "---"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "---"
    return f"{100.0 * x:.{digits}f}"


def alpha_sections(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for key, value in metrics.items():
        if not key.startswith("alpha=") or not isinstance(value, dict):
            continue
        section = value.get("conditional_qed") or value.get("conditional")
        if isinstance(section, dict) and section.get("spearman_rho") is not None:
            rows.append((key, section))
    rows.sort(key=lambda item: float(item[0].split("=", 1)[1]))
    return rows


def passes_quality_gate(section: dict[str, Any]) -> bool:
    return (
        float(section.get("validity", 0.0)) >= 0.99
        and float(section.get("uniqueness", 0.0)) >= 0.90
        and float(section.get("novelty", 0.0)) >= 0.95
    )


def best_qed(metrics: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    sections = alpha_sections(metrics)
    if not sections:
        return "", {}, "missing"
    gated = [(alpha, section) for alpha, section in sections if passes_quality_gate(section)]
    pool = gated or sections
    alpha, section = max(
        pool,
        key=lambda item: (
            float(item[1].get("spearman_rho", -999.0)),
            float(item[1].get("uniqueness", 0.0)),
        ),
    )
    return alpha, section, "pass" if gated else "fail"


def running_names() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "outputs" / "publication_ext").glob("parallel_runner_status*.json"):
        payload = load_json(path)
        if payload.get("state") != "running":
            continue
        for item in payload.get("running", []):
            name = item.get("name")
            if name:
                names.add(str(name))
    return names


def row_for_entry(entry: dict[str, Any], running: set[str]) -> dict[str, Any]:
    name = entry["name"]
    metrics_path = ROOT / entry["output_dir"] / "final_metrics.json"
    metrics = load_json(metrics_path)
    alpha, qed, gate = best_qed(metrics)
    status = "complete_pass" if metrics and gate == "pass" else "complete_fail" if metrics else "pending"
    if name in running and not metrics:
        status = "running_or_incomplete"

    vae_metrics = {}
    depends = entry.get("depends_on") or []
    if depends:
        vae_metrics = load_json(ROOT / str(depends[0]))

    return {
        "status": status,
        "experiment": name,
        "display": entry.get("display", name),
        "purpose": entry.get("purpose", ""),
        "vae_run": entry.get("vae_run", ""),
        "vae_exact_recon": vae_metrics.get("exact_recon"),
        "vae_prior_vun": vae_metrics.get("prior_vun"),
        "best_alpha": alpha,
        "quality_gate": gate,
        "spearman_rho": qed.get("spearman_rho"),
        "mae": qed.get("mae"),
        "slope": qed.get("slope"),
        "validity": qed.get("validity"),
        "uniqueness": qed.get("uniqueness"),
        "novelty": qed.get("novelty"),
        "int_div": qed.get("int_div"),
        "success_0p10": qed.get("success_0p10"),
        "output_dir": entry.get("output_dir", ""),
        "config": entry.get("config", ""),
        "vae_checkpoint": entry.get("vae_checkpoint", ""),
        "latent_cache": entry.get("latent_cache", ""),
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "experiment",
        "display",
        "purpose",
        "vae_run",
        "vae_exact_recon",
        "vae_prior_vun",
        "best_alpha",
        "quality_gate",
        "spearman_rho",
        "mae",
        "slope",
        "validity",
        "uniqueness",
        "novelty",
        "int_div",
        "success_0p10",
        "output_dir",
        "config",
        "vae_checkpoint",
        "latent_cache",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_status(rows: list[dict[str, Any]]) -> None:
    complete = [row for row in rows if str(row["status"]).startswith("complete")]
    pending = [row for row in rows if not str(row["status"]).startswith("complete")]
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    OUT_STATUS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATUS.write_text(
        json.dumps(
            {
                "num_experiments": len(rows),
                "complete": len(complete),
                "pending_or_incomplete": len(pending),
                "minimum_completed_runs_for_downstream_sensitivity": 1,
                "minimum_completed_runs_reached": len(complete) >= 1,
                "status_counts": counts,
                "pending_or_incomplete_entries": pending,
            },
            indent=2,
        )
        + "\n"
    )


def write_tex(rows: list[dict[str, Any]]) -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/collect_vae_drift_results.py",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        "VAE checkpoint & Status & VAE exact (\\%) & $\\rho$ & MAE & U (\\%) & N (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        status = str(row["status"]).replace("_", "\\_")
        lines.append(
            f"{row['display']} & {status} & {pct(row.get('vae_exact_recon'))} & "
            f"{fmt(row.get('spearman_rho'))} & {fmt(row.get('mae'))} & "
            f"{pct(row.get('uniqueness'))} & {pct(row.get('novelty'))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    OUT_TEX.write_text("\n".join(lines) + "\n")


def main() -> None:
    manifest = load_json(MANIFEST)
    entries = [entry for entry in manifest.get("entries", []) if entry.get("group") == "vae_drift_downstream"]
    running = running_names()
    rows = [row_for_entry(entry, running) for entry in entries]
    write_csv(rows)
    write_status(rows)
    write_tex(rows)
    print(f"Collected {len(rows)} downstream VAE drift entries")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_STATUS.relative_to(ROOT)}")
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
