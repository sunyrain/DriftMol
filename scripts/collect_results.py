#!/usr/bin/env python3
"""Collect DriftingMol experiment outputs into publication-ready tables.

The original helper assumed a small fixed matrix of experiments and a single
alpha. This version scans all result roots, selects the best alpha under a
quality gate, and flags protocol issues such as the legacy multi4 binning run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in minimal test envs
    yaml = None


PROP_SECTIONS = [
    ("qed", "conditional_qed", "QED"),
    ("sa_score", "conditional_sa_score", "SA"),
    ("logp", "conditional_logp", "LogP"),
    ("molwt", "conditional_molwt", "MolWt"),
]

EXP_RE = re.compile(r"^exp_(?P<variant>.+)_(?P<condition>qed|multi4|uncond)(?P<suffix>_v2)?$")
PUB_RE = re.compile(r"^pub_(?P<variant>.+?)_(?P<condition>qed|multi4|uncond)(?:_.+)?$")
EXT_RE = re.compile(r"^ext_(?P<variant>.+?)_(?P<condition>qed|multi4|uncond|vae)(?:_.+)?$")
SEED_RE = re.compile(r"(?:^|_)s(?P<seed>\d+)(?:_|$)")
QED_3SEED_VARIANTS = {"F", "A6", "A8", "G4"}


@dataclass
class ExperimentRow:
    status: str
    root: str
    experiment: str
    variant: str
    condition: str
    output_dir: str
    protocol: str = ""
    warning: str = ""
    seed: int | None = None
    manifest_group: str = ""
    alpha: str = ""
    score_key: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def output_dir_aliases(value: str) -> set[str]:
    path = Path(value)
    aliases = {str(path)}
    try:
        aliases.add(str(path.resolve()))
    except Exception:
        pass
    return aliases


def load_running_output_dirs(status_root: Path) -> set[str]:
    running: set[str] = set()
    for status_path in status_root.glob("runner_status*.json"):
        payload = _read_json(status_path)
        if payload.get("state") != "running":
            continue
        entry = payload.get("entry")
        if not isinstance(entry, dict):
            continue
        output_dir = entry.get("output_dir")
        if output_dir:
            running.update(output_dir_aliases(str(output_dir)))
    return running


def apply_running_status(rows: list["ExperimentRow"], status_root: Path) -> None:
    running_dirs = load_running_output_dirs(status_root)
    if not running_dirs:
        return
    for row in rows:
        if row.status == "pending" and output_dir_aliases(row.output_dir) & running_dirs:
            row.status = "running_or_incomplete"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100.0 * float(value):.{digits}f}"


def parse_experiment_name(name: str) -> tuple[str, str]:
    match = EXP_RE.match(name)
    if not match:
        match = PUB_RE.match(name)
    if not match:
        match = EXT_RE.match(name)
    if not match:
        return name, "unknown"
    return match.group("variant"), match.group("condition")


def parse_seed_from_name(name: str) -> int | None:
    match = SEED_RE.search(name)
    if not match:
        return None
    return int(match.group("seed"))


def infer_protocol(output_dir: Path, cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (protocol, warning) for an experiment."""
    root = output_dir.parent.name
    condition = parse_experiment_name(output_dir.name)[1]
    binning = cfg.get("cond_binning", {})
    data_cfg = cfg.get("data", {})
    cfg_cfg = cfg.get("cfg", {})
    prop_indices = data_cfg.get("prop_indices") or []
    positive_mode = cfg_cfg.get("positive_mode", "")
    bin_enabled = bool(binning.get("enabled", False))

    protocol_parts = []
    if condition == "multi4":
        protocol_parts.append("multi4")
    elif condition == "qed":
        protocol_parts.append("qed")
    elif condition == "uncond":
        protocol_parts.append("uncond")
    else:
        protocol_parts.append(condition)

    protocol_parts.append(f"pos={positive_mode or 'unknown'}")
    protocol_parts.append(f"bin={'on' if bin_enabled else 'off'}")
    if root == "final_v2":
        protocol_parts.append("v2")

    warning = ""
    if condition == "multi4" and bin_enabled and len(prop_indices) > 1:
        warning = "legacy_multi4_qed_binning"
    return ", ".join(protocol_parts), warning


def alpha_entries(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for key, val in metrics.items():
        if key.startswith("alpha=") and isinstance(val, dict):
            rows.append((key, val))
    rows.sort(key=lambda item: float(item[0].split("=", 1)[1]))
    return rows


def _passes_gate(section: dict[str, Any], min_validity: float, min_uniqueness: float) -> bool:
    return (
        section.get("validity", 0.0) >= min_validity
        and section.get("uniqueness", 0.0) >= min_uniqueness
        and section.get("novelty", 0.0) >= 0.95
        and section.get("spearman_rho") is not None
    )


def summarize_qed(
    metrics: dict[str, Any],
    min_validity: float,
    min_uniqueness: float,
) -> tuple[str, dict[str, Any], str]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    gated_out: list[tuple[str, dict[str, Any]]] = []
    for alpha, entry in alpha_entries(metrics):
        section = entry.get("conditional_qed") or entry.get("conditional")
        if not isinstance(section, dict):
            continue
        if _passes_gate(section, min_validity, min_uniqueness):
            candidates.append((alpha, section))
        else:
            gated_out.append((alpha, section))

    pool = candidates or gated_out
    if not pool:
        return "", {}, "missing"

    best_alpha, best = max(
        pool,
        key=lambda item: (
            item[1].get("spearman_rho", -999.0),
            item[1].get("uniqueness", 0.0),
        ),
    )
    gate = "pass" if candidates else "fail"
    return best_alpha, dict(best), gate


def summarize_multi4(
    metrics: dict[str, Any],
    min_validity: float,
    min_uniqueness: float,
) -> tuple[str, dict[str, Any], str]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    gated_out: list[tuple[str, dict[str, Any]]] = []

    for alpha, entry in alpha_entries(metrics):
        prop_metrics = []
        for prop_id, section_key, label in PROP_SECTIONS:
            section = entry.get(section_key)
            if not isinstance(section, dict) or section.get("spearman_rho") is None:
                continue
            prop_metrics.append((prop_id, label, section))
        if len(prop_metrics) != len(PROP_SECTIONS):
            continue

        rhos = [s.get("spearman_rho", float("nan")) for _, _, s in prop_metrics]
        uniq = [s.get("uniqueness", float("nan")) for _, _, s in prop_metrics]
        slopes = [s.get("slope", float("nan")) for _, _, s in prop_metrics]
        maes = [s.get("mae", float("nan")) for _, _, s in prop_metrics]
        summary = {
            "avg_spearman_rho": sum(rhos) / len(rhos),
            "min_spearman_rho": min(rhos),
            "avg_uniqueness": sum(uniq) / len(uniq),
            "min_uniqueness": min(uniq),
            "avg_slope": sum(slopes) / len(slopes),
            "avg_mae": sum(maes) / len(maes),
        }
        for prop_id, _, section in prop_metrics:
            summary[f"{prop_id}_rho"] = section.get("spearman_rho")
            summary[f"{prop_id}_mae"] = section.get("mae")
            summary[f"{prop_id}_slope"] = section.get("slope")
            summary[f"{prop_id}_uniqueness"] = section.get("uniqueness")

        passes = all(_passes_gate(s, min_validity, min_uniqueness) for _, _, s in prop_metrics)
        if passes:
            candidates.append((alpha, summary))
        else:
            gated_out.append((alpha, summary))

    pool = candidates or gated_out
    if not pool:
        return "", {}, "missing"

    best_alpha, best = max(
        pool,
        key=lambda item: (
            item[1].get("avg_spearman_rho", -999.0),
            item[1].get("min_uniqueness", 0.0),
        ),
    )
    gate = "pass" if candidates else "fail"
    return best_alpha, dict(best), gate


def summarize_uncond(metrics: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    candidates = []
    for alpha, entry in alpha_entries(metrics):
        section = entry.get("unconditional")
        if isinstance(section, dict):
            candidates.append((alpha, section))
    if not candidates:
        return "", {}, "missing"
    best_alpha, best = max(
        candidates,
        key=lambda item: (
            item[1].get("vun", -999.0),
            item[1].get("uniqueness", 0.0),
        ),
    )
    return best_alpha, dict(best), "pass"


def load_row(output_dir: Path, min_validity: float, min_uniqueness: float) -> ExperimentRow:
    variant, condition = parse_experiment_name(output_dir.name)
    cfg = _read_json(output_dir / "resolved_config.json")
    protocol, warning = infer_protocol(output_dir, cfg)
    row = ExperimentRow(
        status="pending",
        root=output_dir.parent.name,
        experiment=output_dir.name,
        variant=variant,
        condition=condition,
        output_dir=str(output_dir),
        protocol=protocol,
        warning=warning,
        seed=cfg.get("experiment", {}).get("seed"),
        config=cfg,
    )

    metrics_path = output_dir / "final_metrics.json"
    if not metrics_path.exists():
        if (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists():
            row.status = "running_or_incomplete"
        return row

    metrics = _read_json(metrics_path)
    if condition == "multi4":
        alpha, summary, gate = summarize_multi4(metrics, min_validity, min_uniqueness)
        row.score_key = "avg_spearman_rho"
    elif condition == "uncond":
        alpha, summary, gate = summarize_uncond(metrics)
        row.score_key = "vun"
    else:
        alpha, summary, gate = summarize_qed(metrics, min_validity, min_uniqueness)
        row.score_key = "spearman_rho"

    row.status = f"complete_{gate}"
    row.alpha = alpha
    row.metrics = summary
    if gate == "fail":
        row.warning = (row.warning + "; " if row.warning else "") + "quality_gate_failed"
    return row


def manifest_row(entry: dict[str, Any]) -> ExperimentRow:
    output_dir = Path(entry["output_dir"])
    cfg_path = Path(entry.get("config", ""))
    cfg = _read_yaml(cfg_path) if cfg_path else {}
    protocol, warning = infer_protocol(output_dir, cfg)
    variant, condition = parse_experiment_name(output_dir.name)
    seed = cfg.get("experiment", {}).get("seed")
    if seed is None:
        seed = parse_seed_from_name(output_dir.name)
    return ExperimentRow(
        status="pending",
        root=output_dir.parent.name,
        experiment=output_dir.name,
        variant=variant,
        condition=condition,
        output_dir=str(output_dir),
        protocol=protocol,
        warning=warning,
        seed=seed,
        manifest_group=entry.get("group", ""),
        config=cfg,
    )


def apply_manifest_metadata(row: ExperimentRow, entry: dict[str, Any]) -> None:
    row.manifest_group = entry.get("group", "")
    if not row.config:
        cfg_path = Path(entry.get("config", ""))
        row.config = _read_yaml(cfg_path) if cfg_path else {}
    if row.seed is None:
        row.seed = row.config.get("experiment", {}).get("seed")
    if row.seed is None:
        row.seed = parse_seed_from_name(row.experiment)
    if not row.protocol or "unknown" in row.protocol:
        protocol, warning = infer_protocol(Path(row.output_dir), row.config)
        row.protocol = protocol
        if warning and warning not in row.warning:
            row.warning = (row.warning + "; " if row.warning else "") + warning


def find_experiment_dirs(roots: list[Path]) -> list[Path]:
    dirs = []
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith(("exp_", "pub_", "ext_")):
                dirs.append(child)
    return dirs


def load_manifest_entries(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = _read_json(path)
    entries = data.get("entries", [])
    return entries if isinstance(entries, list) else []


def covered_by_canonical_seed42(entry: dict[str, Any]) -> bool:
    """Return True when a publication seed-42 entry is covered by final results."""
    if entry.get("group") != "qed_3seed":
        return False
    output_dir = Path(entry.get("output_dir", ""))
    variant, condition = parse_experiment_name(output_dir.name)
    seed = parse_seed_from_name(output_dir.name)
    if variant not in QED_3SEED_VARIANTS or condition != "qed" or seed != 42:
        return False
    return Path(f"outputs/final/exp_{variant}_qed/final_metrics.json").exists()


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _ci95_radius(std: float | None, n: int) -> float | None:
    if std is None or n < 2:
        return None
    # Student-t two-sided 95% critical values for small-N seed studies.
    tcrit = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(n, 1.96)
    return tcrit * std / math.sqrt(n)


def aggregate_qed_3seed(rows: list[ExperimentRow]) -> list[dict[str, Any]]:
    """Aggregate canonical QED three-seed rows for the main paper table.

    Seed 42 is taken from the canonical `outputs/final` runs unless a completed
    publication seed exists. Additional publication-only sweeps such as z-div
    and paper-faithfulness audits are excluded by manifest group.
    """
    by_variant_seed: dict[tuple[str, int], ExperimentRow] = {}

    def row_priority(row: ExperimentRow) -> tuple[int, float]:
        root_priority = {"seeds": 3, "final": 2}.get(row.root, 1)
        return root_priority, float(row.metrics.get("spearman_rho", -999.0))

    for row in rows:
        if row.condition != "qed" or not row.status.startswith("complete"):
            continue
        if row.seed is None or row.variant not in QED_3SEED_VARIANTS:
            continue
        canonical = row.manifest_group == "qed_3seed" or row.root == "final"
        if not canonical:
            continue
        key = (row.variant, row.seed)
        prev = by_variant_seed.get(key)
        if prev is None or row_priority(row) > row_priority(prev):
            by_variant_seed[key] = row

    grouped: dict[str, list[ExperimentRow]] = {}
    for (variant, _seed), row in by_variant_seed.items():
        grouped.setdefault(variant, []).append(row)

    summaries = []
    for variant, variant_rows in grouped.items():
        variant_rows = sorted(variant_rows, key=lambda r: r.seed or -1)
        rhos = [float(r.metrics["spearman_rho"]) for r in variant_rows if r.metrics.get("spearman_rho") is not None]
        uniq = [float(r.metrics["uniqueness"]) for r in variant_rows if r.metrics.get("uniqueness") is not None]
        maes = [float(r.metrics["mae"]) for r in variant_rows if r.metrics.get("mae") is not None]
        slopes = [float(r.metrics["slope"]) for r in variant_rows if r.metrics.get("slope") is not None]
        rho_mean, rho_std = _mean_std(rhos)
        u_mean, u_std = _mean_std(uniq)
        mae_mean, mae_std = _mean_std(maes)
        slope_mean, slope_std = _mean_std(slopes)
        summaries.append({
            "variant": variant,
            "seeds": ",".join(str(r.seed) for r in variant_rows if r.seed is not None),
            "n": len(variant_rows),
            "rho_mean": rho_mean,
            "rho_std": rho_std,
            "rho_ci95": _ci95_radius(rho_std, len(rhos)),
            "uniqueness_mean": u_mean,
            "uniqueness_std": u_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "slope_mean": slope_mean,
            "slope_std": slope_std,
        })

    summaries.sort(key=lambda item: (item.get("rho_mean") or -999.0), reverse=True)
    return summaries


def row_to_csv(row: ExperimentRow) -> dict[str, Any]:
    base = {
        "status": row.status,
        "manifest_group": row.manifest_group,
        "root": row.root,
        "experiment": row.experiment,
        "variant": row.variant,
        "condition": row.condition,
        "seed": row.seed,
        "alpha": row.alpha,
        "score_key": row.score_key,
        "protocol": row.protocol,
        "warning": row.warning,
        "output_dir": row.output_dir,
    }
    for key, val in sorted(row.metrics.items()):
        base[key] = val
    return base


def write_csv(rows: list[ExperimentRow], path: Path) -> None:
    dicts = [row_to_csv(row) for row in rows]
    fieldnames = sorted({key for item in dicts for key in item.keys()})
    preferred = [
        "status", "manifest_group", "root", "experiment", "variant", "condition", "seed",
        "alpha", "score_key", "spearman_rho", "avg_spearman_rho", "vun",
        "uniqueness", "avg_uniqueness", "min_uniqueness", "validity",
        "novelty", "mae", "avg_mae", "slope", "avg_slope", "protocol",
        "warning", "output_dir",
    ]
    ordered = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, lineterminator="\n")
        writer.writeheader()
        writer.writerows(dicts)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(rows: list[ExperimentRow], path: Path) -> None:
    complete = [r for r in rows if r.status.startswith("complete")]
    pending = [r for r in rows if not r.status.startswith("complete")]
    warnings = [r for r in rows if r.warning]

    qed = [r for r in complete if r.condition == "qed"]
    multi4 = [r for r in complete if r.condition == "multi4"]
    uncond = [r for r in complete if r.condition == "uncond"]

    def by_metric(metric: str):
        return lambda r: float(r.metrics.get(metric, -999.0))

    md = []
    md.append("# DriftingMol Publication Results\n")
    md.append("Generated by `scripts/collect_results.py` from JSON files under `outputs/`.\n")
    md.append("Quality gate defaults: validity >= 0.95, uniqueness >= 0.10, novelty >= 0.95.\n")

    md.append("## QED Conditional\n")
    qed_rows = []
    for r in sorted(qed, key=by_metric("spearman_rho"), reverse=True):
        m = r.metrics
        qed_rows.append([
            r.root,
            r.variant,
            r.alpha,
            _fmt(m.get("spearman_rho")),
            _pct(m.get("uniqueness")),
            _fmt(m.get("mae")),
            _fmt(m.get("slope")),
            r.warning or "-",
        ])
    md.append(_table(["Root", "Variant", "Alpha", "rho", "U%", "MAE", "Slope", "Warning"], qed_rows or [["-", "-", "-", "-", "-", "-", "-", "-"]]))
    md.append("")

    seed_rows = []
    for item in aggregate_qed_3seed(rows):
        seed_rows.append([
            item["variant"],
            item["seeds"] or "-",
            f"{item['n']}/3",
            _fmt(item.get("rho_mean")),
            _fmt(item.get("rho_std")),
            _fmt(item.get("rho_ci95")),
            _pct(item.get("uniqueness_mean")),
            _pct(item.get("uniqueness_std")),
            _fmt(item.get("mae_mean")),
            _fmt(item.get("slope_mean")),
        ])
    if seed_rows:
        md.append("## QED 3-Seed Aggregate\n")
        md.append("Seed 42 uses the canonical `outputs/final` runs when the duplicate publication seed-42 output is absent.\n")
        md.append(_table(
            ["Variant", "Seeds", "Complete", "rho mean", "rho sd", "rho 95% CI", "U mean", "U sd", "MAE mean", "Slope mean"],
            seed_rows,
        ))
        md.append("")

    md.append("## Multi4 Conditional\n")
    m4_rows = []
    for r in sorted(multi4, key=by_metric("avg_spearman_rho"), reverse=True):
        m = r.metrics
        m4_rows.append([
            r.root,
            r.variant,
            r.alpha,
            _fmt(m.get("avg_spearman_rho")),
            _fmt(m.get("qed_rho")),
            _fmt(m.get("sa_score_rho")),
            _fmt(m.get("logp_rho")),
            _fmt(m.get("molwt_rho")),
            _pct(m.get("min_uniqueness")),
            r.warning or "-",
        ])
    md.append(_table(["Root", "Variant", "Alpha", "Avg rho", "QED", "SA", "LogP", "MolWt", "Lowest U%", "Warning"], m4_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]]))
    md.append("")

    md.append("## Unconditional\n")
    u_rows = []
    for r in sorted(uncond, key=by_metric("vun"), reverse=True):
        m = r.metrics
        u_rows.append([
            r.root,
            r.variant,
            r.alpha,
            _fmt(m.get("vun")),
            _pct(m.get("validity")),
            _pct(m.get("uniqueness")),
            _pct(m.get("novelty")),
            _fmt(m.get("qed_mean")),
            _fmt(m.get("fcd"), 2),
        ])
    md.append(_table(["Root", "Variant", "Alpha", "VUN", "V%", "U%", "N%", "QED mean", "FCD"], u_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-"]]))
    md.append("")

    if warnings:
        md.append("## Warnings\n")
        warn_rows = [[r.root, r.experiment, r.warning, r.protocol] for r in warnings]
        md.append(_table(["Root", "Experiment", "Warning", "Protocol"], warn_rows))
        md.append("")

    if pending:
        md.append("## Pending Or Incomplete\n")
        pending_rows = [[r.root, r.experiment, r.status, r.output_dir] for r in pending]
        md.append(_table(["Root", "Experiment", "Status", "Path"], pending_rows))
        md.append("")

    path.write_text("\n".join(md))


def build_status(rows: list[ExperimentRow]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    pending_rows = [r for r in rows if not r.status.startswith("complete")]
    return {
        "num_experiments": len(rows),
        "complete": sum(1 for r in rows if r.status.startswith("complete")),
        "pending_or_incomplete": len(pending_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "pending_or_incomplete_entries": [row_to_csv(r) for r in pending_rows],
        "qed_3seed": aggregate_qed_3seed(rows),
        "warnings": [row_to_csv(r) for r in rows if r.warning],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "outputs/final",
            "outputs/final_phi",
            "outputs/final_v2",
            "outputs/publication/seeds",
            "outputs/publication/audit",
            "outputs/publication/zdiv",
            "outputs/publication/multi4",
        ],
        help="Result roots to scan.",
    )
    parser.add_argument("--out-dir", default="results", help="Directory for generated tables.")
    parser.add_argument(
        "--manifest",
        default="configs/publication/manifest.json",
        help="Publication manifest used to add pending rows for experiments whose output directories do not exist yet.",
    )
    parser.add_argument(
        "--status-root",
        default="outputs/publication",
        help="Directory containing publication runner_status*.json files used to mark just-started runs as active.",
    )
    parser.add_argument("--min-validity", type=float, default=0.95)
    parser.add_argument("--min-uniqueness", type=float, default=0.10)
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_dirs = find_experiment_dirs(roots)
    rows = [
        load_row(path, min_validity=args.min_validity, min_uniqueness=args.min_uniqueness)
        for path in experiment_dirs
    ]
    rows_by_dir = {row.output_dir: row for row in rows}

    manifest_path = Path(args.manifest) if args.manifest else None
    manifest_entries = load_manifest_entries(manifest_path)
    for entry in manifest_entries:
        if covered_by_canonical_seed42(entry):
            continue
        output_dir = str(Path(entry["output_dir"]))
        row = rows_by_dir.get(output_dir)
        if row is None:
            row = manifest_row(entry)
            rows.append(row)
            rows_by_dir[output_dir] = row
        else:
            apply_manifest_metadata(row, entry)

    apply_running_status(rows, Path(args.status_root))

    rows.sort(key=lambda r: (r.condition, r.root, r.variant, r.experiment))

    write_csv(rows, out_dir / "publication_results.csv")
    write_markdown(rows, out_dir / "publication_summary.md")

    status = build_status(rows)
    (out_dir / "publication_status.json").write_text(json.dumps(status, indent=2))

    print(f"Scanned {len(rows)} experiments")
    print(f"Wrote {out_dir / 'publication_summary.md'}")
    print(f"Wrote {out_dir / 'publication_results.csv'}")
    print(f"Wrote {out_dir / 'publication_status.json'}")


if __name__ == "__main__":
    main()
