#!/usr/bin/env python3
"""Summarize archived graph-line experiments as a representation stress note.

This script does not run new experiments. It converts the archived graph VAE
line outputs into a compact diagnostic package so the paper can discuss
representation dependence without promoting graph generation to the main claim.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "archive" / "graph_vae_line" / "outputs"
RAW_VS_REPAIR_JSON = ROOT / "archive" / "graph_vae_line" / "outputs" / "raw_vs_repair_eval.json"
RESULT_JSON = ROOT / "results" / "graph_stress_test.json"
RESULT_TEX = ROOT / "results" / "tables" / "tab_graph_stress.tex"
RESULT_RAW_VS_REPAIR_TEX = ROOT / "results" / "tables" / "tab_graph_raw_vs_repaired.tex"
RESULT_MD = ROOT / "docs" / "GRAPH_STRESS_TEST.md"
RESULTS_CSV = ROOT / "results" / "publication_results.csv"


RUN_SPECS = [
    {
        "family": "Graph VAE prior",
        "run": "vae_v2_kl01",
        "mode": "generation",
        "summary_key": "generation",
        "note": "Valid and novel, but the prior is still sensitive to VAE training.",
    },
    {
        "family": "Graph VAE prior",
        "run": "vae_v3_valence",
        "mode": "generation",
        "summary_key": "generation",
        "note": "Validity reaches 100%, but uniqueness and novelty drop sharply.",
    },
    {
        "family": "Graph drift",
        "run": "drifting_v2kl01_fix2",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "note": "Best unconditional graph-drift quality in the archive.",
    },
    {
        "family": "Decoder drift",
        "run": "e34_decoder_drift_v3",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "note": "Perfect validity, but diversity and novelty fall off.",
    },
    {
        "family": "CFG graph drift",
        "run": "e36_dec_drift_cfg",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "control_prefix": "prop_control",
        "control_target": "QED",
        "note": "Strongest graph control in the archive, yet rho stays low and diversity collapses.",
    },
    {
        "family": "Phi-space drift",
        "run": "E30_phi_space_drift",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "control_prefix": "qed_control",
        "control_target": "QED",
        "note": "A near-null control diagnostic: validity is fine, control is almost absent.",
    },
    {
        "family": "LogP queue",
        "run": "e40_logp_bins_queue",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "control_prefix": "prop_control",
        "control_target": "LogP",
        "note": "Perfect validity, but control remains weak compared with the SELFIES anchor.",
    },
]

FRESH_RUN_SPECS = [
    {
        "family": "Fresh graph QED",
        "run": "e36_dec_drift_cfg_fresh",
        "metrics_path": ROOT / "outputs" / "publication_ext" / "graph_stress" / "e36_dec_drift_cfg_fresh" / "final_metrics.json",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "control_prefix": "prop_control",
        "control_target": "QED",
        "note": "Fresh graph-route QED control under the same V/U/N and Spearman reporting contract.",
    },
    {
        "family": "Fresh graph LogP",
        "run": "e40_logp_bins_queue_fresh",
        "metrics_path": ROOT / "outputs" / "publication_ext" / "graph_stress" / "e40_logp_bins_queue_fresh" / "final_metrics.json",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "control_prefix": "prop_control",
        "control_target": "LogP",
        "note": "Fresh graph-route LogP control for property-transfer comparison.",
    },
    {
        "family": "Fresh graph ablation",
        "run": "e36_no_drift_fresh",
        "metrics_path": ROOT / "outputs" / "publication_ext" / "graph_stress" / "e36_no_drift_fresh" / "final_metrics.json",
        "mode": "generation_uncond",
        "summary_key": "generation_uncond",
        "control_prefix": "prop_control",
        "control_target": "QED",
        "note": "No-drift graph control ablation to isolate the effect of latent drifting.",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def num(value, digits: int = 3) -> str:
    if value in (None, "", "---"):
        return "---"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val):
        return "---"
    return f"{val:.{digits}f}"


def pct(value, digits: int = 1) -> str:
    if value in (None, "", "---"):
        return "---"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val):
        return "---"
    return f"{100.0 * val:.{digits}f}"


def load_publication_anchor() -> dict:
    with RESULTS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if (
            row.get("variant") == "F"
            and row.get("condition") == "qed"
            and row.get("root") == "final"
            and row.get("status", "").startswith("complete")
        ):
            return {
                "family": "SELFIES anchor",
                "run": "DriftingMol (F)",
                "mode": "main paper",
                "control_target": "QED",
                "validity": float(row["validity"]),
                "uniqueness": float(row["uniqueness"]),
                "novelty": float(row["novelty"]),
                "control_rho": float(row["spearman_rho"]),
                "mae": float(row["mae"]),
                "slope": float(row["slope"]),
                "note": "Main-paper anchor for comparison against graph stress results.",
            }
    raise RuntimeError("Could not locate the main QED anchor row in publication_results.csv")


def summarize_run(spec: dict) -> dict:
    metrics_path = Path(spec.get("metrics_path", ARCHIVE_ROOT / spec["run"] / "final_metrics.json"))
    data = load_json(metrics_path)
    gen = data[spec["summary_key"]]
    result = {
        "family": spec["family"],
        "run": spec["run"],
        "mode": spec["mode"],
        "validity": float(gen["validity"]),
        "uniqueness": float(gen["uniqueness"]),
        "novelty": float(gen["novelty"]),
        "note": spec["note"],
    }

    prefix = spec.get("control_prefix")
    if prefix:
        control_rows = []
        for key, value in data.items():
            if not key.startswith(f"{prefix}_"):
                continue
            rho = value.get("spearman_rho")
            if rho is None:
                continue
            control_rows.append(
                {
                    "setting": key.split("_", 2)[-1],
                    "rho": float(rho),
                    "gap": float(value.get("qed_gap", value.get("prop_gap", float("nan")))),
                }
            )
        control_rows.sort(key=lambda item: item["rho"])
        if control_rows:
            best = max(control_rows, key=lambda item: item["rho"])
            result["control_target"] = spec.get("control_target", "")
            result["control_best_rho"] = best["rho"]
            result["control_best_setting"] = best["setting"]
            result["control_rho_min"] = control_rows[0]["rho"]
            result["control_rho_max"] = control_rows[-1]["rho"]
            gaps = [item["gap"] for item in control_rows if not math.isnan(item["gap"])]
            if gaps:
                result["control_gap_min"] = min(gaps)
                result["control_gap_max"] = max(gaps)
    return result


def available_run_specs() -> list[dict]:
    specs = list(RUN_SPECS)
    for spec in FRESH_RUN_SPECS:
        metrics_path = Path(spec["metrics_path"])
        if metrics_path.exists():
            specs.append(spec)
    return specs


def summarize_raw_vs_repair() -> list[dict]:
    if not RAW_VS_REPAIR_JSON.exists():
        return []
    payload = load_json(RAW_VS_REPAIR_JSON)
    rows: list[dict] = []
    family_map = {
        "e38_drift_v2": "Graph drift",
        "e34_decoder_drift_v3": "Decoder drift",
    }
    for run, temps in payload.items():
        if not isinstance(temps, dict):
            continue
        for temp_key, entry in temps.items():
            if not isinstance(entry, dict):
                continue
            raw = entry.get("raw", {})
            repair = entry.get("repair", {})
            try:
                temp = float(str(temp_key).removeprefix("temp_"))
            except ValueError:
                continue
            rows.append(
                {
                    "family": family_map.get(run, run),
                    "run": run,
                    "temp": temp,
                    "raw_validity": float(raw.get("validity", float("nan"))),
                    "repair_validity": float(repair.get("validity", float("nan"))),
                    "raw_uniqueness": float(raw.get("uniqueness", float("nan"))),
                    "repair_uniqueness": float(repair.get("uniqueness", float("nan"))),
                    "raw_novelty": float(raw.get("novelty", float("nan"))),
                    "repair_novelty": float(repair.get("novelty", float("nan"))),
                    "raw_vun": float(raw.get("vun", float("nan"))),
                    "repair_vun": float(repair.get("vun", float("nan"))),
                    "boost_validity": float(entry.get("repair_boost_validity", float("nan"))),
                    "boost_uniqueness": float(entry.get("repair_boost_uniqueness", float("nan"))),
                }
            )
    rows.sort(key=lambda row: (row["run"], row["temp"]))
    return rows


def write_json(payload: dict) -> None:
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {RESULT_JSON}")


def write_tex(rows: list[dict]) -> None:
    RESULT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/summarize_graph_stress.py",
        "\\begin{tabular}{l l l c c c c l}",
        "\\toprule",
        "Family & Run & Mode & Validity (\\%) & Uniqueness (\\%) & Novelty (\\%) & Best $\\rho$ & Note \\\\",
        "\\midrule",
    ]
    for row in rows:
        best_rho = row.get("control_best_rho", row.get("control_rho"))
        lines.append(
            f"{row['family']} & {row['run']} & {row['mode']} & "
            f"{pct(row.get('validity'))} & {pct(row.get('uniqueness'))} & {pct(row.get('novelty'))} & "
            f"{num(best_rho)} & {row['note']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    RESULT_TEX.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULT_TEX}")


def write_raw_vs_repair_tex(rows: list[dict]) -> None:
    if not rows:
        return
    RESULT_RAW_VS_REPAIR_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/summarize_graph_stress.py",
        "\\begin{tabular}{l c c c c c c c}",
        "\\toprule",
        "Family & $\\tau$ & Raw valid. & Repaired valid. & Raw uniq. & Repaired uniq. & $\\Delta$valid. & $\\Delta$uniq. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['family']} & {num(row['temp'], 1)} & {pct(row['raw_validity'])} & {pct(row['repair_validity'])} & "
            f"{pct(row['raw_uniqueness'])} & {pct(row['repair_uniqueness'])} & {num(row['boost_validity'])} & {num(row['boost_uniqueness'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    RESULT_RAW_VS_REPAIR_TEX.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULT_RAW_VS_REPAIR_TEX}")


def write_md(rows: list[dict], anchor: dict, raw_vs_repair: list[dict]) -> None:
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Graph Representation Stress Test",
        "",
        "Updated: 2026-05-15 UTC",
        "",
        "This note summarizes archived graph VAE-line experiments. It is a diagnostic",
        "artifact, not a claim that graph generation should replace the SELFIES main",
        "track.",
        "",
        "## Main-Paper Anchor",
        "",
        f"- SELFIES DriftingMol (F): validity {pct(anchor['validity'])}%, uniqueness {pct(anchor['uniqueness'])}%, "
        f"novelty {pct(anchor['novelty'])}%, QED $\\rho$ {num(anchor['control_rho'])}, MAE {num(anchor['mae'])}",
        "",
        "## Archived Generation Quality",
        "",
        "| Family | Run | Mode | Validity | Uniqueness | Novelty | Note |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | `{row['run']}` | {row['mode']} | "
            f"{pct(row.get('validity'))}% | {pct(row.get('uniqueness'))}% | {pct(row.get('novelty'))}% | {row['note']} |"
        )

    lines += [
        "",
        "## Control Summary",
        "",
        "| Family | Run | Target | Best $\\rho$ | $\\rho$ range | Gap range | Note |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if "control_best_rho" not in row:
            continue
        rho_range = f"{num(row.get('control_rho_min'))} to {num(row.get('control_rho_max'))}"
        gap_min = row.get("control_gap_min")
        gap_max = row.get("control_gap_max")
        gap_range = "n/a"
        if gap_min is not None and gap_max is not None:
            gap_range = f"{num(gap_min)} to {num(gap_max)}"
        lines.append(
            f"| {row['family']} | `{row['run']}` | {row.get('control_target', '')} | "
            f"{num(row.get('control_best_rho'))} | {rho_range} | {gap_range} | {row['note']} |"
        )
    lines.append(
        f"| {anchor['family']} | `{anchor['run']}` | {anchor.get('control_target', '')} | "
        f"{num(anchor.get('control_rho'))} | {num(anchor.get('control_rho'))} to {num(anchor.get('control_rho'))} | n/a | {anchor['note']} |"
    )

    if raw_vs_repair:
        lines += [
            "",
            "## Raw-vs-Repaired Decoding",
            "",
            "This diagnostic separates sanitization repair from actual control. Repair always",
            "restores validity to 100%, but uniqueness gains are small and can turn slightly",
            "negative at higher temperatures.",
            "",
            "| Family | Temperature | Raw validity | Repaired validity | Raw uniqueness | Repaired uniqueness | $\\Delta$validity | $\\Delta$uniqueness |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in raw_vs_repair:
            lines.append(
                f"| {row['family']} | {num(row['temp'], 1)} | {pct(row['raw_validity'])}% | {pct(row['repair_validity'])}% | "
                f"{pct(row['raw_uniqueness'])}% | {pct(row['repair_uniqueness'])}% | {num(row['boost_validity'])} | {num(row['boost_uniqueness'])} |"
            )

    lines += [
        "",
        "## Reading",
        "",
        "1. Graph validity is not the main issue. The harder problem is stable diversity",
        "   and meaningful target control.",
        "2. Stronger graph control variants still sit well below the SELFIES anchor in $\\rho$.",
        "3. For the AAAI draft, graph results are best used as a limitation / diagnostic",
        "   appendix, not as a replacement for the current SELFIES story.",
        "",
        "## Recommendation",
        "",
        "- Keep SELFIES as the main method in the submission package.",
        "- If graph work is continued, run one clean QM9 graph-control pass and one",
        "  destructive graph ablation; do not expand into a full second method line.",
    ]
    RESULT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULT_MD}")


def main() -> None:
    anchor = load_publication_anchor()
    rows = [summarize_run(spec) for spec in available_run_specs()]
    raw_vs_repair = summarize_raw_vs_repair()
    payload = {
        "source": str(ARCHIVE_ROOT),
        "anchor": anchor,
        "rows": rows,
        "raw_vs_repair": raw_vs_repair,
        "summary": {
            "main_point": "Graph representation quality is usable as a diagnostic, but control remains weak relative to the SELFIES anchor.",
            "paper_policy": "Keep the current SELFIES main track and treat graph results as a limitation / appendix artifact.",
            "repair_note": "Raw-vs-repaired decoding is summarized; repair restores validity while leaving control weak.",
        },
    }
    write_json(payload)
    write_tex([*rows, anchor])
    write_raw_vs_repair_tex(raw_vs_repair)
    write_md(rows, anchor, raw_vs_repair)


if __name__ == "__main__":
    main()
