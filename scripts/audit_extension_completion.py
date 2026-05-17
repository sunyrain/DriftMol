#!/usr/bin/env python3
"""Audit completion of the post-draft AAAI extension work packages.

This audit intentionally covers work that the main publication audit does not:
matched baselines, destructive drift ablations, graph stress tests, and VAE
sensitivity. By default it writes the audit and exits 0 so it can be used while
work is still open. Use --strict to make open required items return non-zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_MD = RESULTS / "extension_completion_audit.md"
OUT_JSON = RESULTS / "extension_completion_status.json"


@dataclass
class Check:
    requirement: str
    evidence: str
    status: str
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def check_publication_package() -> Check:
    audit = RESULTS / "publication_completion_audit.md"
    text = audit.read_text(errors="replace") if audit.exists() else ""
    ok = "Overall: PASS" in text
    return Check(
        "Current main submission package remains audited",
        f"{rel(audit)} contains Overall: PASS" if ok else f"{rel(audit)} missing or not PASS",
        "PASS" if ok else "OPEN",
    )


def check_graph_diagnostic() -> Check:
    paths = [
        ROOT / "scripts" / "summarize_graph_stress.py",
        ROOT / "docs" / "GRAPH_STRESS_TEST.md",
        RESULTS / "graph_stress_test.json",
        RESULTS / "tables" / "tab_graph_stress.tex",
    ]
    missing = [rel(p) for p in paths if not exists_nonempty(p)]
    data = load_json(RESULTS / "graph_stress_test.json")
    rows = data.get("rows", [])
    ok = not missing and len(rows) >= 3
    evidence = (
        f"{len(rows)} graph stress rows; artifacts present"
        if ok
        else f"missing={missing or 'none'}, rows={len(rows)}"
    )
    return Check("Graph diagnostic snapshot exists", evidence, "PASS" if ok else "OPEN")


def check_graph_followup_plan() -> Check:
    plan = ROOT / "docs" / "GRAPH_STRESS_EXECUTION_PLAN.md"
    text = plan.read_text(errors="replace") if plan.exists() else ""
    required_phrases = [
        "Fresh graph QED control reproduction",
        "Fresh graph LogP control reproduction",
        "Graph destructive drift ablation",
        "Completion Gate",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    ok = exists_nonempty(plan) and not missing
    evidence = f"{rel(plan)} present; missing_sections={missing}"
    return Check("Graph stress follow-up execution plan exists", evidence, "PASS" if ok else "OPEN", required=False)


def check_graph_stress_manifest() -> Check:
    manifest = ROOT / "configs" / "publication_ext" / "graph_stress_manifest.json"
    namespace_plan = ROOT / "docs" / "GRAPH_NAMESPACE_ADAPTER_PLAN.md"
    payload = load_json(manifest)
    entries = payload.get("entries", [])
    preconditions = payload.get("preconditions", [])
    archived = payload.get("archived_diagnostics", [])
    ok = (
        exists_nonempty(manifest)
        and exists_nonempty(namespace_plan)
        and isinstance(entries, list)
        and len(entries) >= 5
        and isinstance(preconditions, list)
        and len(preconditions) >= 4
        and isinstance(archived, list)
        and len(archived) >= 2
    )
    evidence = (
        f"entries={len(entries) if isinstance(entries, list) else 0}; "
        f"preconditions={len(preconditions) if isinstance(preconditions, list) else 0}; "
        f"archived_diagnostics={len(archived) if isinstance(archived, list) else 0}; "
        f"namespace_plan={namespace_plan.exists()}"
    )
    return Check("Graph stress prepared manifest exists", evidence, "PASS" if ok else "OPEN", required=False)


def check_graph_archive_launchability() -> Check:
    script = ROOT / "scripts" / "audit_graph_archive_launchability.py"
    status_path = RESULTS / "graph_archive_launchability_status.json"
    audit_path = RESULTS / "graph_archive_launchability_audit.md"
    status = load_json(status_path)
    blockers = status.get("namespace", {}).get("blockers", [])
    missing_artifacts = status.get("missing_required_artifacts", [])
    archived_metrics = [
        row for row in status.get("archived_diagnostic_runs", [])
        if row.get("metrics_exists")
    ]
    ok = exists_nonempty(script) and exists_nonempty(status_path) and exists_nonempty(audit_path)
    evidence = (
        "complete={complete}; missing_artifacts={missing}; blockers={blockers}; archived_metrics={metrics}".format(
            complete=bool(status.get("complete")),
            missing=len(missing_artifacts),
            blockers=len(blockers),
            metrics=len(archived_metrics),
        )
        if status
        else "launchability status not written"
    )
    return Check("Graph archive launchability preflight exists", evidence, "PASS" if ok else "OPEN", required=False)


def check_full_graph_stress() -> Check:
    fig = ROOT / "docs" / "figures" / "fig_graph_bottleneck.pdf"
    status = RESULTS / "graph_stress_full_status.json"
    if exists_nonempty(status):
        payload = load_json(status)
        complete = bool(payload.get("complete"))
        fig = ROOT / str(payload.get("figure_pdf", ""))
        return Check(
            "Full graph representation stress test is complete",
            f"{rel(status)} complete={complete}; figure_exists={exists_nonempty(fig)}",
            "PASS" if complete else "OPEN",
        )
    return Check(
        "Full graph representation stress test is complete",
        f"diagnostic exists, but {rel(fig)} and {rel(status)} are not present",
        "OPEN",
    )


def check_destructive_infra() -> Check:
    manifest_path = ROOT / "configs" / "publication_ext" / "manifest.json"
    manifest = load_json(manifest_path)
    entries = manifest.get("entries", [])
    destructive = [entry for entry in entries if entry.get("group") == "destructive_drift"]
    missing_configs = []
    bad_commands = []
    for entry in destructive:
        cfg = entry.get("config", "")
        if not cfg or not (ROOT / cfg).exists():
            missing_configs.append(cfg or entry.get("name", "<missing config>"))
        if cfg and f"--config {cfg}" not in entry.get("command", ""):
            bad_commands.append(entry.get("name", cfg))

    drift_source = (ROOT / "src" / "drifting" / "drift_latent_phi.py").read_text(errors="replace")
    train_source = (ROOT / "src" / "train" / "train_selfies_cfg.py").read_text(errors="replace")
    source_ok = (
        "attraction_scale" in drift_source
        and "repulsion_scale" in drift_source
        and "drift_attraction_scale" in train_source
        and "drift_repulsion_scale" in train_source
    )
    ok = len(destructive) >= 7 and not missing_configs and not bad_commands and source_ok
    evidence = (
        f"{len(destructive)} runnable destructive configs; source scale hooks present"
        if ok
        else f"entries={len(destructive)}, missing_configs={missing_configs}, bad_commands={bad_commands}, source_ok={source_ok}"
    )
    return Check("Destructive drift ablation infrastructure is ready", evidence, "PASS" if ok else "OPEN")


def check_parallel_runner() -> Check:
    script = ROOT / "scripts" / "run_manifest_parallel.py"
    status = load_json(ROOT / "outputs" / "publication_ext" / "parallel_runner_status.json")
    ok = exists_nonempty(script) and status.get("state") in {"dry_run", "running", "completed"}
    evidence = (
        f"script exists; last_state={status.get('state')}; devices={','.join(status.get('devices', []))}"
        if status
        else "parallel runner script exists but no status file found"
    )
    return Check("Parallel extension runner is available", evidence, "PASS" if ok else "OPEN", required=False)


def check_destructive_results() -> Check:
    status = load_json(RESULTS / "destructive_ablation_status.json")
    complete = int(status.get("complete", 0) or 0)
    total = int(status.get("num_experiments", 0) or 0)
    pending = int(status.get("pending_or_incomplete", 0) or 0)
    reached = bool(status.get("minimum_completed_runs_reached", False))
    counts = status.get("status_counts", {})
    ok = complete >= 7 and total >= 7 and pending == 0 and reached
    return Check(
        "Destructive drift ablation results are complete",
        f"complete={complete}/{total}, pending_or_incomplete={pending}, status_counts={counts}, minimum_completed_runs_reached={reached}",
        "PASS" if ok else "OPEN",
    )


def check_matched_baselines() -> Check:
    data = load_json(RESULTS / "matched_baselines_qed.json")
    baselines = data.get("baselines", {})
    table = RESULTS / "tables" / "tab_qed_matched_baselines.tex"
    n_baselines = len(baselines) if isinstance(baselines, dict) else 0
    ok = n_baselines >= 3 and exists_nonempty(table)
    evidence = f"baselines={n_baselines}; table_exists={exists_nonempty(table)}"
    status = "PASS" if ok else ("PARTIAL" if n_baselines > 0 else "OPEN")
    return Check("Protocol-matched baselines cover at least three non-trivial methods", evidence, status)


def check_vae_sensitivity() -> Check:
    status = load_json(RESULTS / "vae_sensitivity_status.json")
    complete = int(status.get("complete", 0) or 0)
    total = int(status.get("num_experiments", 0) or 0)
    pending = int(status.get("pending_or_incomplete", 0) or 0)
    configs = sorted((ROOT / "configs" / "publication_ext" / "vae_sensitivity").glob("*.yaml"))
    ok = len(configs) >= 4 and total >= 4 and complete >= total and pending == 0
    evidence = (
        f"configs={len(configs)}, complete={complete}/{total}, pending_or_incomplete={pending}"
        if status
        else f"configs={len(configs)}, no results/vae_sensitivity_status.json"
    )
    return Check("VAE sensitivity study is complete", evidence, "PASS" if ok else "OPEN")


def check_vae_drift_queue() -> Check:
    manifest = load_json(ROOT / "configs" / "publication_ext" / "vae_drift_manifest.json")
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    missing_configs = []
    for entry in entries:
        if not isinstance(entry, dict):
            missing_configs.append("<invalid entry>")
            continue
        cfg = entry.get("config", "")
        if not cfg or not (ROOT / cfg).exists():
            missing_configs.append(cfg or str(entry.get("name", "<missing config>")))
    collector = ROOT / "scripts" / "collect_vae_drift_results.py"
    status = RESULTS / "vae_drift_downstream_status.json"
    table = RESULTS / "tables" / "tab_vae_drift_downstream.tex"
    ok = len(entries) >= 4 and not missing_configs and exists_nonempty(collector) and exists_nonempty(status) and exists_nonempty(table)
    evidence = (
        f"entries={len(entries)}, collector={exists_nonempty(collector)}, status={exists_nonempty(status)}, table={exists_nonempty(table)}"
        if ok
        else f"entries={len(entries)}, missing_configs={missing_configs}, collector={exists_nonempty(collector)}, status={exists_nonempty(status)}, table={exists_nonempty(table)}"
    )
    return Check("Downstream VAE-drift queue and collector are ready", evidence, "PASS" if ok else "OPEN", required=False)


def check_vae_drift_results() -> Check:
    status = load_json(RESULTS / "vae_drift_downstream_status.json")
    complete = int(status.get("complete", 0) or 0)
    total = int(status.get("num_experiments", 0) or 0)
    pending = int(status.get("pending_or_incomplete", 0) or 0)
    table = RESULTS / "tables" / "tab_vae_drift_downstream.tex"
    ok = complete >= total and total >= 4 and pending == 0 and exists_nonempty(table)
    evidence = f"complete={complete}/{total}, pending_or_incomplete={pending}, table_exists={exists_nonempty(table)}"
    return Check("Downstream VAE-drift results are complete for all alternative checkpoints", evidence, "PASS" if ok else "OPEN")


def check_generalization_queue() -> Check:
    manifest = load_json(ROOT / "configs" / "publication_ext" / "generalization_manifest.json")
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    missing_configs = []
    for entry in entries:
        if not isinstance(entry, dict):
            missing_configs.append("<invalid entry>")
            continue
        cfg = entry.get("config", "")
        if not cfg or not (ROOT / cfg).exists():
            missing_configs.append(cfg or str(entry.get("name", "<missing config>")))
    collector = ROOT / "scripts" / "collect_generalization_results.py"
    status = RESULTS / "generalization_status.json"
    table = RESULTS / "tables" / "tab_generalization.tex"
    launcher_files = sorted((ROOT / "outputs" / "publication_ext").glob("generalization_launcher_gpu*.pid"))
    postprocess = ROOT / "outputs" / "publication_ext" / "generalization_postprocess.pid"
    ok = (
        len(entries) >= 4
        and not missing_configs
        and exists_nonempty(collector)
        and exists_nonempty(status)
        and exists_nonempty(table)
        and len(launcher_files) >= 4
        and exists_nonempty(postprocess)
    )
    evidence = (
        f"entries={len(entries)}, missing_configs={missing_configs}, collector={exists_nonempty(collector)}, "
        f"status={exists_nonempty(status)}, table={exists_nonempty(table)}, launchers={len(launcher_files)}, "
        f"postprocess={exists_nonempty(postprocess)}"
    )
    return Check("Generalization queue and collector are ready", evidence, "PASS" if ok else "OPEN", required=False)


def check_generalization_results() -> Check:
    status = load_json(RESULTS / "generalization_status.json")
    complete = int(status.get("complete", 0) or 0)
    total = int(status.get("num_experiments", 0) or 0)
    pending = int(status.get("pending_or_incomplete", 0) or 0)
    table = RESULTS / "tables" / "tab_generalization.tex"
    ok = complete >= total and total >= 4 and pending == 0 and exists_nonempty(table)
    evidence = f"complete={complete}/{total}, pending_or_incomplete={pending}, table_exists={exists_nonempty(table)}"
    return Check("Generalization results are complete", evidence, "PASS" if ok else "OPEN")


def check_reviewer_extra_queue() -> Check:
    manifest = load_json(ROOT / "configs" / "publication_ext" / "reviewer_extra_manifest.json")
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    missing_configs = []
    for entry in entries:
        if not isinstance(entry, dict):
            missing_configs.append("<invalid entry>")
            continue
        cfg = entry.get("config", "")
        if not cfg or not (ROOT / cfg).exists():
            missing_configs.append(cfg or str(entry.get("name", "<missing config>")))
    collector = ROOT / "scripts" / "collect_reviewer_extra_results.py"
    watcher = ROOT / "scripts" / "watch_reviewer_extra_postprocess.py"
    status = RESULTS / "reviewer_extra_status.json"
    table = RESULTS / "tables" / "tab_reviewer_extra.tex"
    launcher_files = sorted((ROOT / "outputs" / "publication_ext").glob("reviewer_extra_launcher_gpu*.pid"))
    postprocess = ROOT / "outputs" / "publication_ext" / "reviewer_extra_postprocess.pid"
    ok = (
        len(entries) >= 4
        and not missing_configs
        and exists_nonempty(collector)
        and exists_nonempty(watcher)
        and exists_nonempty(status)
        and exists_nonempty(table)
        and len(launcher_files) >= 4
        and exists_nonempty(postprocess)
    )
    evidence = (
        f"entries={len(entries)}, missing_configs={missing_configs}, collector={exists_nonempty(collector)}, "
        f"watcher={exists_nonempty(watcher)}, status={exists_nonempty(status)}, table={exists_nonempty(table)}, "
        f"launchers={len(launcher_files)}, postprocess={exists_nonempty(postprocess)}"
    )
    return Check("Reviewer-extra queue and collector are ready", evidence, "PASS" if ok else "OPEN", required=False)


def check_reviewer_extra_results() -> Check:
    status = load_json(RESULTS / "reviewer_extra_status.json")
    complete = int(status.get("complete", 0) or 0)
    total = int(status.get("num_experiments", 0) or 0)
    pending = int(status.get("pending_or_incomplete", 0) or 0)
    table = RESULTS / "tables" / "tab_reviewer_extra.tex"
    ok = complete >= total and total >= 4 and pending == 0 and exists_nonempty(table)
    evidence = f"complete={complete}/{total}, pending_or_incomplete={pending}, table_exists={exists_nonempty(table)}"
    return Check("Reviewer-extra bin/seed robustness results are complete", evidence, "PASS" if ok else "OPEN")


def check_trained_baselines() -> Check:
    status = load_json(RESULTS / "trained_baseline_status.json")
    table = RESULTS / "tables" / "tab_trained_baseline_qed.tex"
    manifest = load_json(ROOT / "configs" / "publication_ext" / "baseline_manifest.json")
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    complete = int(status.get("complete", 0) or 0)
    total = int(status.get("num_experiments", 0) or 0)
    three_seed_complete = bool(status.get("three_seed_complete", False))
    ok = len(entries) >= 3 and total >= 3 and three_seed_complete and exists_nonempty(table)
    evidence = (
        f"manifest_entries={len(entries)}, complete={complete}/{total}, "
        f"three_seed_complete={three_seed_complete}, table_exists={exists_nonempty(table)}"
    )
    return Check("Trained property-guidance baseline has a three-seed summary", evidence, "PASS" if ok else "OPEN")


def check_extension_checklist() -> Check:
    checklist = ROOT / "docs" / "EXTENSION_EXECUTION_CHECKLIST.md"
    csv_path = RESULTS / "destructive_ablation.csv"
    tex_path = RESULTS / "tables" / "tab_destructive_ablation.tex"
    paths = [checklist, csv_path, tex_path]
    missing = [rel(path) for path in paths if not exists_nonempty(path)]
    return Check(
        "Extension execution checklist and destructive result artifacts exist",
        "all present" if not missing else f"missing={missing}",
        "PASS" if not missing else "OPEN",
    )


def build_checks() -> list[Check]:
    return [
        check_publication_package(),
        check_graph_diagnostic(),
        check_graph_followup_plan(),
        check_graph_stress_manifest(),
        check_graph_archive_launchability(),
        check_full_graph_stress(),
        check_destructive_infra(),
        check_parallel_runner(),
        check_destructive_results(),
        check_matched_baselines(),
        check_vae_sensitivity(),
        check_vae_drift_queue(),
        check_vae_drift_results(),
        check_generalization_queue(),
        check_generalization_results(),
        check_reviewer_extra_queue(),
        check_reviewer_extra_results(),
        check_trained_baselines(),
        check_extension_checklist(),
    ]


def write_outputs(checks: list[Check]) -> bool:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    required = [check for check in checks if check.required]
    complete = all(check.ok for check in required)
    lines = [
        "# DriftingMol Extension Completion Audit",
        "",
        "Objective: complete the post-draft AAAI extension work packages without",
        "confusing the stable main submission package with unfinished follow-up",
        "experiments.",
        "",
        "| Requirement | Evidence | Status |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {check.requirement} | {check.evidence} | {check.status} |")
    lines += [
        "",
        f"Overall extension completion: {'PASS' if complete else 'OPEN'}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    OUT_JSON.write_text(
        json.dumps(
            {
                "complete": complete,
                "checks": [
                    {
                        "requirement": check.requirement,
                        "evidence": check.evidence,
                        "status": check.status,
                        "required": check.required,
                    }
                    for check in checks
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero while required items are open.")
    args = parser.parse_args()

    checks = build_checks()
    complete = write_outputs(checks)
    print(f"Wrote {rel(OUT_MD)}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Overall extension completion: {'PASS' if complete else 'OPEN'}")
    if args.strict and not complete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
