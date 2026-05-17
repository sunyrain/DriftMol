#!/usr/bin/env python3
"""Audit whether the DriftingMol publication package is complete.

This is a gatekeeper script, not a result collector.  It checks concrete
artifacts against the active publication objective and exits non-zero until the
experiment, manuscript, table, and figure deliverables are all present.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_manuscript_benchmark import update_text
from scripts.export_latex_tables import (
    MULTI4_LABELS,
    MULTI4_ORDER,
    QED_LABELS,
    QED_ORDER,
    SEED_LABELS,
    fnum,
    fpct,
)

RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
PAPER = ROOT / "docs" / "PAPER_DRAFT.md"
AAAI_PAPER = ROOT / "docs" / "PAPER_AAAI.tex"
AAAI_STYLE = ROOT / "docs" / "aaai2026.sty"
AAAI_BST = ROOT / "docs" / "aaai2026.bst"
AAAI_BIB = ROOT / "docs" / "references_aaai.bib"
PAPER_BUILD = ROOT / "docs" / "PAPER_BUILD.md"
README = ROOT / "README.md"
FULL_RESULTS = ROOT / "docs" / "FULL_RESULTS.md"
MANIFEST = ROOT / "configs" / "publication" / "manifest.json"
STATUS = RESULTS / "publication_status.json"
CSV = RESULTS / "publication_results.csv"
INFERENCE_BENCHMARK = RESULTS / "inference_benchmark.json"

QED_VARIANTS = {"F", "A6", "A8", "G4"}
AUDIT_NAMES = {
    "pub_P1_paper_tau_batch_lambda_qed_s42",
    "pub_P2_paper_tau_fixed_lambda_qed_s42",
    "pub_P3_no_cfg_qed_s42",
    "pub_P4_no_zdiv_qed_s42",
    "pub_P5_y_only_norm_qed_s42",
    "pub_P6_no_cross_norm_qed_s42",
}
ZDIV_NAMES = {
    "pub_G4_qed_zdiv0p0_s42",
    "pub_G4_qed_zdiv0p5_s42",
    "pub_G4_qed_zdiv1p0_s42",
    "pub_G4_qed_zdiv2p0_s42",
    "pub_G4_qed_zdiv4p0_s42",
}
QED_SEED_NAMES = {
    f"pub_{variant}_qed_s{seed}"
    for variant in sorted(QED_VARIANTS)
    for seed in (42, 43, 44)
}
REQUIRED_MANIFEST_NAMES = QED_SEED_NAMES | ZDIV_NAMES | AUDIT_NAMES | {"pub_G4_multi4_v2_s42"}


@dataclass
class Check:
    requirement: str
    evidence: str
    ok: bool


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def complete(row: dict[str, str]) -> bool:
    return row.get("status", "").startswith("complete")


def complete_pass(row: dict[str, str]) -> bool:
    return row.get("status", "") == "complete_pass"


def rows_by_experiment(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("experiment", ""): row for row in rows if row.get("experiment")}


def check_artifacts(status: dict) -> list[Check]:
    paths = [
        RESULTS / "publication_summary.md",
        RESULTS / "publication_results.csv",
        RESULTS / "publication_status.json",
        RESULTS / "tables" / "tab_qed_main.tex",
        RESULTS / "tables" / "tab_multi4_v2.tex",
        RESULTS / "tables" / "tab_qed_3seed.tex",
    ]
    missing = [display_path(p) for p in paths if not p.exists()]
    empty = [
        f"{display_path(p)}={p.stat().st_size}B"
        for p in paths
        if p.exists() and p.stat().st_size == 0
    ]
    return [
        Check(
            "Result CSV, Markdown summary, status JSON, and LaTeX tables exist and are non-empty",
            (
                ("missing: " + ", ".join(missing) if missing else "")
                + ("; " if missing and empty else "")
                + ("empty: " + ", ".join(empty) if empty else "")
                if missing or empty
                else "all expected result artifacts present and non-empty"
            ),
            not missing and not empty and bool(status),
        )
    ]


def _table_data_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = set()
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if " & " not in stripped or not stripped.endswith("\\\\"):
            continue
        if stripped.startswith("Variant &") or stripped.startswith("Model / control &"):
            continue
        lines.add(stripped)
    return lines


def _expected_qed_table_lines(rows: list[dict[str, str]]) -> set[str]:
    by_variant = {
        r["variant"]: r
        for r in rows
        if r.get("condition") == "qed"
        and r.get("status", "").startswith("complete")
        and r.get("variant") in QED_ORDER
        and not r.get("manifest_group")
        and r.get("root", "final") in {"final", "final_phi"}
    }
    expected = set()
    for variant in QED_ORDER:
        row = by_variant.get(variant)
        if row is None:
            continue
        alpha = row.get("alpha", "").replace("alpha=", "")
        expected.add(
            f"{QED_LABELS[variant]} & {alpha} & {fnum(row.get('spearman_rho'))} & "
            f"{fnum(row.get('slope'))} & {fpct(row.get('uniqueness'))} & "
            f"{fnum(row.get('mae'))} \\\\"
        )
    return expected


def _expected_multi4_table_lines(rows: list[dict[str, str]]) -> set[str]:
    by_variant = {
        r["variant"]: r
        for r in rows
        if r.get("condition") == "multi4"
        and r.get("status", "").startswith("complete")
        and (r.get("root") == "final_v2" or (r.get("root") == "multi4" and r.get("variant") == "G4"))
    }
    expected = set()
    for variant in MULTI4_ORDER:
        row = by_variant.get(variant)
        if row is None:
            continue
        alpha = row.get("alpha", "").replace("alpha=", "")
        expected.add(
            f"{MULTI4_LABELS[variant]} & {alpha} & {fnum(row.get('qed_rho'))} & "
            f"{fnum(row.get('sa_score_rho'))} & {fnum(row.get('logp_rho'))} & "
            f"{fnum(row.get('molwt_rho'))} & {fnum(row.get('avg_spearman_rho'))} & "
            f"{fpct(row.get('min_uniqueness'))} \\\\"
        )
    return expected


def _expected_qed_seed_table_lines(status: dict) -> set[str]:
    expected = set()
    for item in status.get("qed_3seed", []):
        ci = item.get("rho_ci95")
        ci_text = "---" if ci is None else fnum(ci)
        label = SEED_LABELS.get(item.get("variant"), item.get("variant"))
        expected.add(
            f"{label} & {item.get('seeds', '')} & {item.get('n', 0)} & "
            f"{fnum(item.get('rho_mean'))} & {ci_text} \\\\"
        )
    return expected


def check_latex_table_sync(rows: list[dict[str, str]], status: dict) -> list[Check]:
    table_specs = [
        ("tab_qed_main.tex", _expected_qed_table_lines(rows)),
        ("tab_multi4_v2.tex", _expected_multi4_table_lines(rows)),
        ("tab_qed_3seed.tex", _expected_qed_seed_table_lines(status)),
    ]
    problems = []
    for filename, expected in table_specs:
        path = RESULTS / "tables" / filename
        actual = _table_data_lines(path)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            problems.append(f"{filename} missing {len(missing)} expected rows")
        if extra:
            problems.append(f"{filename} has {len(extra)} unexpected rows")
    return [
        Check(
            "Generated LaTeX tables are synchronized to CSV/status sources",
            "; ".join(problems) if problems else "QED main, fair multi4, and QED 3-seed table rows match source artifacts",
            not problems,
        )
    ]


def check_manifest(status: dict) -> list[Check]:
    manifest = load_json(MANIFEST)
    entries = manifest.get("entries", [])
    names = {entry.get("name", "") for entry in entries}
    missing_names = sorted(REQUIRED_MANIFEST_NAMES - names)
    missing_configs = []
    command_mismatches = []
    for entry in entries:
        config = entry.get("config", "")
        config_path = ROOT / config
        if not config or not config_path.exists():
            missing_configs.append(config or f"{entry.get('name', '<unnamed>')}:<missing config field>")
        command = entry.get("command", "")
        if config and f"--config {config}" not in command:
            command_mismatches.append(entry.get("name", config))
    pending = int(status.get("pending_or_incomplete", -1))
    return [
        Check(
            "Publication manifest exists and defines every required experiment entry",
            (
                f"{len(entries)} manifest entries; missing: " + ", ".join(missing_names)
                if missing_names
                else f"{len(entries)} required manifest entries present"
            ),
            len(entries) == len(REQUIRED_MANIFEST_NAMES) and not missing_names,
        ),
        Check(
            "Publication manifest entries reference existing config files and matching commands",
            (
                ("missing configs: " + ", ".join(missing_configs) if missing_configs else "")
                + ("; " if missing_configs and command_mismatches else "")
                + ("command mismatches: " + ", ".join(command_mismatches) if command_mismatches else "")
                if missing_configs or command_mismatches
                else "all manifest configs exist and commands reference their config"
            ),
            not missing_configs and not command_mismatches,
        ),
        Check(
            "No required publication experiment is pending or incomplete",
            f"pending_or_incomplete={pending}",
            pending == 0,
        ),
    ]


def check_qed_seeds(status: dict) -> list[Check]:
    rows = {row.get("variant"): row for row in status.get("qed_3seed", [])}
    missing = []
    for variant in sorted(QED_VARIANTS):
        n = int(rows.get(variant, {}).get("n", 0))
        if n != 3:
            missing.append(f"{variant}={n}/3")
    return [
        Check(
            "F/A6/A8/G4 QED variants have complete 3-seed coverage",
            ", ".join(missing) if missing else "all key variants have 3/3 seeds",
            not missing,
        )
    ]


def check_experiment_groups(rows: list[dict[str, str]]) -> list[Check]:
    by_exp = rows_by_experiment(rows)

    zdiv_missing = [
        name for name in sorted(ZDIV_NAMES)
        if not complete(by_exp.get(name, {}))
    ]
    audit_missing = [
        name for name in sorted(AUDIT_NAMES)
        if not complete(by_exp.get(name, {}))
    ]
    multi4 = by_exp.get("pub_G4_multi4_v2_s42", {})
    return [
        Check(
            "z-diversity Pareto sweep is complete",
            "missing/incomplete: " + ", ".join(zdiv_missing) if zdiv_missing else "all z-div points complete",
            not zdiv_missing,
        ),
        Check(
            "Fair G4 multi4 publication run is complete and passes the quality gate",
            multi4.get("status", "missing"),
            complete_pass(multi4),
        ),
        Check(
            "Paper-faithfulness audit P1-P6 is complete",
            "missing/incomplete: " + ", ".join(audit_missing) if audit_missing else "P1-P6 complete",
            not audit_missing,
        ),
    ]


def check_inference_benchmark() -> list[Check]:
    bench = load_json(INFERENCE_BENCHMARK)
    if not bench:
        return [
            Check(
                "Inference throughput and 1-NFE claims have benchmark evidence",
                f"missing or unreadable {INFERENCE_BENCHMARK.relative_to(ROOT)}",
                False,
            )
        ]

    nfe = bench.get("nfe", bench.get("generation_protocol", {}).get("nfe"))
    generator_rate = float(bench.get("generator_mol_per_s", 0) or 0)
    end_to_end_rate = float(bench.get("end_to_end_mol_per_s", 0) or 0)
    ok = nfe == 1 and generator_rate > 0 and end_to_end_rate > 0
    return [
        Check(
            "Inference throughput and 1-NFE claims have benchmark evidence",
            f"nfe={nfe}, generator_mol_per_s={generator_rate:.1f}, end_to_end_mol_per_s={end_to_end_rate:.1f}",
            ok,
        )
    ]


def check_benchmark_manuscript_sync() -> list[Check]:
    bench = load_json(INFERENCE_BENCHMARK)
    if not bench:
        return [
            Check(
                "Manuscript computational-cost table is synchronized to benchmark JSON",
                f"missing {INFERENCE_BENCHMARK.relative_to(ROOT)}",
                False,
            )
        ]
    if not PAPER.exists():
        return [
            Check(
                "Manuscript computational-cost table is synchronized to benchmark JSON",
                f"missing {PAPER.relative_to(ROOT)}",
                False,
            )
        ]
    paper_text = PAPER.read_text(errors="replace")
    try:
        synchronized = update_text(paper_text, bench) == paper_text
    except Exception as exc:
        return [
            Check(
                "Manuscript computational-cost table is synchronized to benchmark JSON",
                f"could not validate sync: {exc}",
                False,
            )
        ]
    return [
        Check(
            "Manuscript computational-cost table is synchronized to benchmark JSON",
            "benchmark rates already reflected in manuscript" if synchronized else "paper differs from benchmark JSON",
            synchronized,
        )
    ]


def check_manuscript_and_figures() -> list[Check]:
    paper_text = PAPER.read_text(errors="replace") if PAPER.exists() else ""
    required_figures = {
        "fig1_main": "fig:main",
        "fig2_qed_ablation": "fig:qed_ablation",
        "fig3_multi4_v2": "fig:multi4_v2",
        "fig4_qed_seed_ci": "fig:qed_seed_ci",
        "fig5_zdiv_pareto": "fig:zdiv_pareto",
    }
    missing_files = []
    invalid_files = []
    missing_figure_refs = []
    for stem, label in required_figures.items():
        missing_stem = False
        for suffix in ("pdf", "png"):
            path = FIGURES / f"{stem}.{suffix}"
            if not path.exists():
                missing_stem = True
            elif path.stat().st_size < 1024:
                invalid_files.append(f"{stem}.{suffix}={path.stat().st_size}B")
        if missing_stem:
            missing_files.append(stem)
        if f"{stem}.pdf" not in paper_text:
            missing_figure_refs.append(f"{stem}.pdf include")
        if f"\\label{{{label}}}" not in paper_text:
            missing_figure_refs.append(f"{label} label")
        if f"\\ref{{{label}}}" not in paper_text:
            missing_figure_refs.append(f"{label} ref")

    tables = ["tab:qed_main", "tab:multi4", "tab:qed_3seed"]
    missing_tables = [label for label in tables if f"\\label{{{label}}}" not in paper_text]
    missing_table_refs = [label for label in tables if f"\\ref{{{label}}}" not in paper_text]
    table_evidence_parts = []
    if missing_tables:
        table_evidence_parts.append("missing labels: " + ", ".join(missing_tables))
    if missing_table_refs:
        table_evidence_parts.append("missing refs: " + ", ".join(missing_table_refs))
    return [
        Check(
            "Manuscript exists and references required final result tables",
            "; ".join(table_evidence_parts) if table_evidence_parts else "required table labels and refs present",
            PAPER.exists() and not missing_tables and not missing_table_refs,
        ),
        Check(
            "All final paper figures exist as non-empty PDF and PNG files",
            (
                ("missing: " + ", ".join(missing_files) if missing_files else "")
                + ("; " if missing_files and invalid_files else "")
                + ("invalid: " + ", ".join(invalid_files) if invalid_files else "")
                if missing_files or invalid_files
                else "all required figure files present and non-empty"
            ),
            not missing_files and not invalid_files,
        ),
        Check(
            "Manuscript references all final paper figures",
            "missing refs: " + ", ".join(missing_figure_refs) if missing_figure_refs else "all required figure includes, labels, and refs present",
            not missing_figure_refs,
        ),
    ]


def check_aaai_source_metadata() -> list[Check]:
    if not AAAI_PAPER.exists():
        return [
            Check(
                "AAAI manuscript source exists with final author metadata",
                f"missing {display_path(AAAI_PAPER)}",
                False,
            )
        ]
    text = AAAI_PAPER.read_text(errors="replace")
    required = {
        "title": "DriftingMol: Decoder-Coupled Drift for One-Pass Property-Conditional Molecular Generation",
        "authors": "Jiangjie Qiu, Yijun Li, Wentao Li, Xiaonan Wang",
        "corresponding author": "Corresponding author.",
        "affiliation": "Beijing Key Laboratory of Artificial Intelligence for Advanced Chemical Engineering Materials",
        "aaai style": "\\usepackage[draft]{aaai2026}",
    }
    missing = [name for name, marker in required.items() if marker not in text]
    return [
        Check(
            "AAAI manuscript source exists with final author metadata",
            "missing markers: " + ", ".join(missing) if missing else "title, authors, corresponding author, affiliation, and AAAI style present",
            not missing,
        )
    ]


def check_aaai_manuscript_and_figures() -> list[Check]:
    paper_text = AAAI_PAPER.read_text(errors="replace") if AAAI_PAPER.exists() else ""
    required_figures = {
        "fig1_main": "fig:main",
        "fig2_qed_ablation": "fig:qed-ablation",
        "fig3_multi4_v2": "fig:multi4",
        "fig4_qed_seed_ci": "fig:qed-seed-ci",
        "fig5_zdiv_pareto": "fig:zdiv-pareto",
    }
    missing_figure_refs = []
    for stem, label in required_figures.items():
        if stem not in paper_text:
            missing_figure_refs.append(f"{stem} include")
        if f"\\label{{{label}}}" not in paper_text:
            missing_figure_refs.append(f"{label} label")
        if f"\\ref{{{label}}}" not in paper_text:
            missing_figure_refs.append(f"{label} ref")

    tables = ["tab:qed-main", "tab:qed-diversity", "tab:qed-3seed", "tab:multi4"]
    missing_tables = [label for label in tables if f"\\label{{{label}}}" not in paper_text]
    missing_table_refs = [label for label in tables if f"\\ref{{{label}}}" not in paper_text]
    table_evidence_parts = []
    if not AAAI_PAPER.exists():
        table_evidence_parts.append(f"missing {display_path(AAAI_PAPER)}")
    if missing_tables:
        table_evidence_parts.append("missing labels: " + ", ".join(missing_tables))
    if missing_table_refs:
        table_evidence_parts.append("missing refs: " + ", ".join(missing_table_refs))
    return [
        Check(
            "AAAI source references required final result tables",
            "; ".join(table_evidence_parts) if table_evidence_parts else "AAAI table labels and refs present",
            AAAI_PAPER.exists() and not missing_tables and not missing_table_refs,
        ),
        Check(
            "AAAI source references all final paper figures",
            "missing refs: " + ", ".join(missing_figure_refs) if missing_figure_refs else "AAAI figure includes, labels, and refs present",
            AAAI_PAPER.exists() and not missing_figure_refs,
        ),
    ]


def check_aaai_build_prerequisites() -> list[Check]:
    engines = [name for name in ("latexmk", "pdflatex", "tectonic") if shutil.which(name)]
    missing = []
    for path in (AAAI_PAPER, AAAI_STYLE, AAAI_BST, AAAI_BIB, PAPER_BUILD):
        if not path.exists():
            missing.append(display_path(path))
    if not engines:
        missing.append("TeX engine: latexmk/pdflatex/tectonic")
    if PAPER_BUILD.exists():
        build_text = PAPER_BUILD.read_text(errors="replace")
        for marker in ("PAPER_AAAI.tex", "aaai2026.sty", "aaai2026.bst"):
            if marker not in build_text:
                missing.append(f"{marker} build note")
    return [
        Check(
            "AAAI LaTeX build prerequisites are documented and available",
            (
                "missing: " + ", ".join(missing)
                if missing
                else f"build notes present; TeX engine={engines[0]}; aaai2026 style/bst/bib present"
            ),
            not missing,
        )
    ]


def _rate_tokens(value: float) -> set[str]:
    rounded_exact = int(round(float(value)))
    rounded_hundred = int(round(float(value) / 100.0) * 100)
    tokens = set()
    for val in {rounded_exact, rounded_hundred}:
        plain = f"{val:,}"
        tokens.add(plain)
        tokens.add(plain.replace(",", "{,}"))
    return tokens


def check_aaai_benchmark_claims() -> list[Check]:
    bench = load_json(INFERENCE_BENCHMARK)
    if not bench:
        return [
            Check(
                "AAAI source reflects inference benchmark throughput",
                f"missing {display_path(INFERENCE_BENCHMARK)}",
                False,
            )
        ]
    if not AAAI_PAPER.exists():
        return [
            Check(
                "AAAI source reflects inference benchmark throughput",
                f"missing {display_path(AAAI_PAPER)}",
                False,
            )
        ]
    text = AAAI_PAPER.read_text(errors="replace")
    missing = []
    generator_tokens = _rate_tokens(float(bench["generator_mol_per_s"]))
    end_to_end_tokens = _rate_tokens(float(bench["end_to_end_mol_per_s"]))
    if not any(token in text for token in generator_tokens):
        missing.append("generator throughput")
    if not any(token in text for token in end_to_end_tokens):
        missing.append("end-to-end throughput")
    num_samples = int(bench.get("num_samples", 0))
    if num_samples:
        sample_tokens = {f"{num_samples:,}", f"{num_samples:,}".replace(",", "{,}")}
        if not any(token in text for token in sample_tokens):
            missing.append("benchmark sample count")
    return [
        Check(
            "AAAI source reflects inference benchmark throughput",
            "missing markers: " + ", ".join(missing) if missing else "generator/end-to-end throughput and sample count reflected in AAAI source",
            not missing,
        )
    ]


def check_aaai_bibliography() -> list[Check]:
    missing_files = [display_path(p) for p in (AAAI_PAPER, AAAI_BIB) if not p.exists()]
    if missing_files:
        return [
            Check(
                "AAAI citations resolve against references_aaai.bib",
                "missing: " + ", ".join(missing_files),
                False,
            )
        ]
    paper_text = AAAI_PAPER.read_text(errors="replace")
    bib_text = AAAI_BIB.read_text(errors="replace")
    cite_groups = re.findall(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}", paper_text)
    cites = {key.strip() for group in cite_groups for key in group.split(",") if key.strip()}
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
    missing_cites = sorted(cites - bib_keys)
    missing_bibliography_command = "references_aaai" not in paper_text
    evidence_parts = []
    if missing_cites:
        evidence_parts.append("missing bib entries: " + ", ".join(missing_cites))
    if missing_bibliography_command:
        evidence_parts.append("missing references_aaai bibliography command")
    return [
        Check(
            "AAAI citations resolve against references_aaai.bib",
            "; ".join(evidence_parts) if evidence_parts else f"{len(cites)} cited keys resolved by references_aaai.bib",
            not missing_cites and not missing_bibliography_command,
        )
    ]


def _sn_jnl_resolution_mode() -> str | None:
    if (ROOT / "sn-jnl.cls").exists():
        return "repo root"
    if (PAPER.parent / "sn-jnl.cls").exists():
        return f"{PAPER.parent.name}/"
    kpsewhich = shutil.which("kpsewhich")
    if not kpsewhich:
        return None
    proc = subprocess.run(
        [kpsewhich, "sn-jnl.cls"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and bool(proc.stdout.strip()):
        return "kpsewhich"
    return None


def _build_notes_include_texinputs_for_local_class(mode: str | None) -> bool:
    if mode != f"{PAPER.parent.name}/":
        return True
    if not PAPER_BUILD.exists():
        return False
    text = PAPER_BUILD.read_text(encoding="utf-8")
    texinputs_tokens = [
        f"TEXINPUTS={PAPER.parent.name}//:",
        f"TEXINPUTS={PAPER.parent.name}:",
    ]
    return any(token in text for token in texinputs_tokens)


def check_manuscript_build_prerequisites() -> list[Check]:
    engines = [name for name in ("pdflatex", "latexmk", "tectonic") if shutil.which(name)]
    class_mode = _sn_jnl_resolution_mode()
    missing = []
    if not PAPER.exists():
        missing.append(display_path(PAPER))
    if not PAPER_BUILD.exists():
        missing.append(display_path(PAPER_BUILD))
    if not engines:
        missing.append("TeX engine: pdflatex/latexmk/tectonic")
    if not class_mode:
        missing.append("sn-jnl.cls")
    if not _build_notes_include_texinputs_for_local_class(class_mode):
        missing.append(f"TEXINPUTS={PAPER.parent.name}//: build command")
    return [
        Check(
            "Manuscript LaTeX build prerequisites are documented and available",
            (
                "missing: " + ", ".join(missing)
                if missing
                else f"build notes present; TeX engine={engines[0]}; sn-jnl.cls resolved via {class_mode}"
            ),
            not missing,
        )
    ]


def check_documentation_consistency() -> list[Check]:
    missing = [display_path(p) for p in (README, FULL_RESULTS) if not p.exists()]
    if missing:
        return [
            Check(
                "Repository result documentation uses the current publication protocol",
                "missing: " + ", ".join(missing),
                False,
            )
        ]

    readme = README.read_text(errors="replace")
    full_results = FULL_RESULTS.read_text(errors="replace")
    stale_patterns = {
        "old_temperature_tau_10": "τ=10",
        "old_temperature_grid": "{1,5,10,20}",
        "old_gru_vae_label": "SELFIES GRU β-VAE",
        "old_readme_multi4_value": "Multi-4-property | Single-τ | **ρ̄ = 0.387**",
    }
    stale = [
        name
        for name, pattern in stale_patterns.items()
        if pattern in readme or pattern in full_results
    ]
    required = {
        "README links generated publication summary": "results/publication_summary.md" in readme,
        "README labels fair multi4 v2": "Multi-4-property v2" in readme,
        "FULL_RESULTS marked legacy snapshot": "Legacy Static Snapshot" in full_results,
        "FULL_RESULTS includes fair v2 table": "Table 4b: Fair Multi-property v2" in full_results,
    }
    missing_required = [name for name, ok in required.items() if not ok]
    ok = not stale and not missing_required
    evidence_parts = []
    if stale:
        evidence_parts.append("stale patterns: " + ", ".join(stale))
    if missing_required:
        evidence_parts.append("missing markers: " + ", ".join(missing_required))
    return [
        Check(
            "Repository result documentation uses the current publication protocol",
            "; ".join(evidence_parts) if evidence_parts else "README and FULL_RESULTS point to generated publication tables and fair multi4 v2",
            ok,
        )
    ]


def check_manuscript_bibliography() -> list[Check]:
    if not PAPER.exists():
        return [
            Check(
                "Manuscript citations and bibliography are internally consistent",
                f"missing {display_path(PAPER)}",
                False,
            )
        ]

    paper_text = PAPER.read_text(errors="replace")
    cite_groups = re.findall(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}", paper_text)
    cites = {key.strip() for group in cite_groups for key in group.split(",") if key.strip()}
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", paper_text))
    missing_bibitems = sorted(cites - bibitems)
    unused_bibitems = sorted(bibitems - cites)
    evidence_parts = []
    if missing_bibitems:
        evidence_parts.append("missing bibitems: " + ", ".join(missing_bibitems))
    if unused_bibitems:
        evidence_parts.append("unused bibitems: " + ", ".join(unused_bibitems))
    return [
        Check(
            "Manuscript citations and bibliography are internally consistent",
            "; ".join(evidence_parts) if evidence_parts else f"{len(cites)} cited keys all resolved",
            not missing_bibitems and not unused_bibitems,
        )
    ]


def check_tests(run_tests: bool) -> list[Check]:
    if not run_tests:
        return [
            Check(
                "Unit tests have been run for this audit",
                "not run; pass --run-tests for live evidence",
                False,
            )
        ]
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    ran_match = re.search(r"Ran (\d+) tests?", proc.stdout)
    if proc.returncode == 0:
        evidence = f"OK ({ran_match.group(1)} tests)" if ran_match else "OK"
    else:
        failure_match = re.search(r"(FAILED \([^)]+\)|ERROR|FAIL)", proc.stdout)
        evidence = failure_match.group(1) if failure_match else (
            proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else f"exit={proc.returncode}"
        )
    return [
        Check(
            "Unit test suite passes",
            evidence,
            proc.returncode == 0,
        )
    ]


def render(checks: list[Check]) -> str:
    lines = [
        "# Publication Completion Audit",
        "",
        "Objective: read and organize the repository, run the missing publication experiments in parallel, fully use compute resources, finish experiments, manuscript writing, and paper figures.",
        "",
        "| Requirement | Evidence | Status |",
        "|---|---|---|",
    ]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        evidence = check.evidence.replace("|", "\\|")
        lines.append(f"| {check.requirement} | {evidence} | {status} |")
    lines.append("")
    if all(c.ok for c in checks):
        lines.append("Overall: PASS. The publication package satisfies the audited completion gate.")
    else:
        lines.append("Overall: FAIL. The publication package is not complete.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests as part of the audit.")
    parser.add_argument(
        "--write",
        default=str(RESULTS / "publication_completion_audit.md"),
        help="Write the audit Markdown to this path. Use '-' to skip writing.",
    )
    args = parser.parse_args()

    status = load_json(STATUS)
    rows = load_rows(CSV)
    checks: list[Check] = []
    checks.extend(check_artifacts(status))
    checks.extend(check_latex_table_sync(rows, status))
    checks.extend(check_manifest(status))
    checks.extend(check_qed_seeds(status))
    checks.extend(check_experiment_groups(rows))
    checks.extend(check_inference_benchmark())
    checks.extend(check_aaai_source_metadata())
    checks.extend(check_aaai_manuscript_and_figures())
    checks.extend(check_aaai_build_prerequisites())
    checks.extend(check_aaai_benchmark_claims())
    checks.extend(check_aaai_bibliography())
    checks.extend(check_benchmark_manuscript_sync())
    checks.extend(check_manuscript_and_figures())
    checks.extend(check_manuscript_build_prerequisites())
    checks.extend(check_documentation_consistency())
    checks.extend(check_manuscript_bibliography())
    checks.extend(check_tests(args.run_tests))

    text = render(checks)
    print(text, end="")
    if args.write != "-":
        out = Path(args.write)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out.relative_to(ROOT)}")

    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
