#!/usr/bin/env python3
"""Audit same-backbone generative baseline completion."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/publication_ext/generative_baselines_manifest.json"
SUMMARY = ROOT / "results/generative_baselines_qed.json"
TABLE = ROOT / "results/tables/tab_generative_baselines_qed.tex"
AUDIT_MD = ROOT / "results/generative_baselines_audit.md"
STATUS_JSON = ROOT / "results/generative_baselines_status.json"

EXPECTED_METHODS = {"cvae", "gan", "diffusion", "flow_matching"}
EXPECTED_SEEDS = {42, 43, 44}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def check() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = load_json(MANIFEST)
    entries = manifest.get("entries", []) if isinstance(manifest.get("entries"), list) else []
    rows.append(
        {
            "requirement": "Manifest exists and has 12 entries",
            "evidence": f"{rel(MANIFEST)} entries={len(entries)}",
            "ok": MANIFEST.exists() and len(entries) == 12,
        }
    )

    seen_methods: set[str] = set()
    seen_seeds: set[int] = set()
    missing_outputs: list[str] = []
    bad_commands: list[str] = []
    for entry in entries:
        command = str(entry.get("command", ""))
        output_dir = ROOT / str(entry.get("output_dir", ""))
        final_path = output_dir / "final_metrics.json"
        method = None
        seed = None
        parts = command.split()
        if "--method" in parts:
            method = parts[parts.index("--method") + 1]
            seen_methods.add(method)
        if "--seed" in parts:
            try:
                seed = int(parts[parts.index("--seed") + 1])
                seen_seeds.add(seed)
            except (ValueError, IndexError):
                pass
        if method is None or method not in EXPECTED_METHODS or seed is None or seed not in EXPECTED_SEEDS:
            bad_commands.append(str(entry.get("name", "")))
        if not final_path.exists() or final_path.stat().st_size == 0:
            missing_outputs.append(rel(final_path))

    rows.append(
        {
            "requirement": "Manifest covers four families and three seeds",
            "evidence": f"methods={sorted(seen_methods)}, seeds={sorted(seen_seeds)}",
            "ok": seen_methods == EXPECTED_METHODS and seen_seeds == EXPECTED_SEEDS and not bad_commands,
        }
    )
    rows.append(
        {
            "requirement": "Every manifest output has final_metrics.json",
            "evidence": "all present" if not missing_outputs else ", ".join(missing_outputs),
            "ok": not missing_outputs and len(entries) == 12,
        }
    )

    summary = load_json(SUMMARY)
    result_rows = summary.get("results", [])
    aggregates = summary.get("aggregates", [])
    rows.append(
        {
            "requirement": "Summary JSON has 12 result rows",
            "evidence": f"{rel(SUMMARY)} rows={len(result_rows) if isinstance(result_rows, list) else 'invalid'}",
            "ok": isinstance(result_rows, list) and len(result_rows) == 12,
        }
    )

    aggregate_ok = isinstance(aggregates, list) and len(aggregates) == 4
    aggregate_evidence = []
    if isinstance(aggregates, list):
        for item in aggregates:
            aggregate_evidence.append(f"{item.get('method')} n={item.get('n')}")
            aggregate_ok = aggregate_ok and item.get("method") in EXPECTED_METHODS and item.get("n") == 3
            metrics = item.get("metrics", {})
            for key in ["spearman_rho", "mae", "uniqueness", "novelty", "int_div"]:
                aggregate_ok = aggregate_ok and metrics.get(key, {}).get("mean") is not None
    rows.append(
        {
            "requirement": "Aggregates report n=3 with core metrics for each family",
            "evidence": ", ".join(aggregate_evidence) if aggregate_evidence else "missing aggregates",
            "ok": aggregate_ok,
        }
    )

    table_text = TABLE.read_text() if TABLE.exists() else ""
    table_ok = TABLE.exists() and TABLE.stat().st_size > 0
    for label in [
        "Conditional latent VAE",
        "Conditional latent WGAN-GP",
        "Conditional latent DDPM",
        "Conditional latent Flow Matching",
    ]:
        table_ok = table_ok and label in table_text
    rows.append(
        {
            "requirement": "LaTeX table exists and includes all four families",
            "evidence": rel(TABLE) if table_ok else "missing or incomplete table",
            "ok": table_ok,
        }
    )

    overall = all(row["ok"] for row in rows)
    status = {
        "overall": "PASS" if overall else "OPEN",
        "num_checks": len(rows),
        "passed": sum(1 for row in rows if row["ok"]),
        "failed": [row for row in rows if not row["ok"]],
    }
    return rows, status


def write_outputs(rows: list[dict[str, Any]], status: dict[str, Any]) -> None:
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generative Baseline Audit",
        "",
        "| Requirement | Evidence | Status |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['requirement']} | {row['evidence']} | {'PASS' if row['ok'] else 'OPEN'} |"
        )
    lines += ["", f"Overall: {status['overall']}"]
    AUDIT_MD.write_text("\n".join(lines) + "\n")
    STATUS_JSON.write_text(json.dumps(status, indent=2) + "\n")


def main() -> int:
    rows, status = check()
    write_outputs(rows, status)
    print(f"Wrote {rel(AUDIT_MD)}")
    print(f"Wrote {rel(STATUS_JSON)}")
    print(f"Overall generative baseline audit: {status['overall']}")
    return 0 if status["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
