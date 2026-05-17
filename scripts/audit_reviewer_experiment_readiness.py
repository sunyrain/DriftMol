#!/usr/bin/env python3
"""Audit readiness of reviewer-facing experiment planning and queues."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "reviewer_experiment_readiness_status.json"
OUT_MD = ROOT / "results" / "reviewer_experiment_readiness_audit.md"


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def read_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def pid_alive(pid_file: str) -> bool:
    path = ROOT / pid_file
    if not path.exists():
        return False
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def manifest_count(rel: str) -> int:
    payload = read_json(rel)
    entries = payload.get("entries", [])
    return len(entries) if isinstance(entries, list) else 0


def manifest_summary(rel: str) -> dict[str, Any]:
    payload = read_json(rel)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    group_counts: dict[str, int] = {}
    missing_configs = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        group = str(entry.get("group", ""))
        group_counts[group] = group_counts.get(group, 0) + 1
        config = entry.get("config")
        if not isinstance(config, str) or not (ROOT / config).exists():
            missing_configs.append(str(entry.get("name", "<unnamed>")))
    return {
        "entries": len(entries),
        "group_counts": group_counts,
        "missing_configs": missing_configs,
    }


def status_row(requirement: str, evidence: str, passed: bool, note: str = "") -> dict[str, Any]:
    return {
        "requirement": requirement,
        "evidence": evidence,
        "status": "PASS" if passed else "OPEN",
        "note": note,
    }


def build_status() -> dict[str, Any]:
    faith = read_json("results/drifting_faithfulness_status.json")
    faithful_runs = read_json("results/faithful_drifting_status.json")
    core_runner = read_json("outputs/reviewer_faithful/core_status.json")
    ext = read_json("results/extension_completion_status.json")
    destructive = read_json("results/destructive_ablation_status.json")
    vae = read_json("results/vae_sensitivity_status.json")
    vae_drift = read_json("results/vae_drift_downstream_status.json")
    generalization = read_json("results/generalization_status.json")
    reviewer_extra = read_json("results/reviewer_extra_status.json")
    next_wave = read_json("results/next_wave_status.json")
    next_wave_runner = read_json("outputs/publication_ext/parallel_runner_status_next_wave_gpu0.json")
    trained_baselines = read_json("results/trained_baseline_status.json")
    graph_launchability = read_json("results/graph_archive_launchability_status.json")
    faithful_manifest = manifest_summary("configs/reviewer_faithful/manifest.json")
    baseline_manifest = manifest_summary("configs/publication_ext/baseline_manifest.json")
    vae_drift_manifest = manifest_summary("configs/publication_ext/vae_drift_manifest.json")
    generalization_manifest = manifest_summary("configs/publication_ext/generalization_manifest.json")
    reviewer_extra_manifest = manifest_summary("configs/publication_ext/reviewer_extra_manifest.json")
    next_wave_manifest = manifest_summary("configs/publication_ext/next_wave_manifest.json")
    graph_stress_manifest = read_json("configs/publication_ext/graph_stress_manifest.json")
    faithful_core_complete = faithful_runs.get("faithful_core_complete") is True
    faithful_allocation_complete = (
        faithful_runs.get("groups", {}).get("faithful_allocation", {}).get("pending", 1) == 0
        and faithful_runs.get("groups", {}).get("faithful_allocation", {}).get("total", 0) == 6
    )
    deferred_launcher_ok = (
        pid_alive("outputs/reviewer_faithful/deferred_faithful_core_launcher.pid")
        or faithful_core_complete
        or core_runner.get("state") == "completed"
    )
    vae_drift_pid_files = [
        "outputs/publication_ext/vae_drift_launcher_gpu0.pid",
        "outputs/publication_ext/vae_drift_launcher_gpu1.pid",
        "outputs/publication_ext/vae_drift_launcher_gpu2.pid",
        "outputs/publication_ext/vae_drift_launcher_gpu3.pid",
    ]
    vae_drift_launchers_alive = sum(1 for path in vae_drift_pid_files if pid_alive(path))
    vae_drift_runner_statuses = [
        "outputs/publication_ext/parallel_runner_status_vae_drift_gpu0.json",
        "outputs/publication_ext/parallel_runner_status_vae_drift_gpu1.json",
        "outputs/publication_ext/parallel_runner_status_vae_drift_gpu2.json",
        "outputs/publication_ext/parallel_runner_status_vae_drift_gpu3.json",
    ]
    vae_drift_automation_present = (
        vae_drift_launchers_alive >= 1
        or any(exists(path) for path in vae_drift_runner_statuses)
        or int(vae_drift.get("complete", 0) or 0) >= 1
    )
    vae_drift_postprocess_ok = (
        pid_alive("outputs/publication_ext/vae_drift_postprocess.pid")
        or int(vae_drift.get("complete", 0) or 0) >= 4
    )
    generalization_pid_files = [
        "outputs/publication_ext/generalization_launcher_gpu0.pid",
        "outputs/publication_ext/generalization_launcher_gpu1.pid",
        "outputs/publication_ext/generalization_launcher_gpu2.pid",
        "outputs/publication_ext/generalization_launcher_gpu3.pid",
    ]
    generalization_launchers_alive = sum(1 for path in generalization_pid_files if pid_alive(path))
    generalization_postprocess_ok = (
        pid_alive("outputs/publication_ext/generalization_postprocess.pid")
        or int(generalization.get("complete", 0) or 0) >= 4
    )
    reviewer_extra_pid_files = [
        "outputs/publication_ext/reviewer_extra_launcher_gpu0.pid",
        "outputs/publication_ext/reviewer_extra_launcher_gpu1.pid",
        "outputs/publication_ext/reviewer_extra_launcher_gpu2.pid",
        "outputs/publication_ext/reviewer_extra_launcher_gpu3.pid",
    ]
    reviewer_extra_launchers_alive = sum(1 for path in reviewer_extra_pid_files if pid_alive(path))
    reviewer_extra_postprocess_ok = (
        pid_alive("outputs/publication_ext/reviewer_extra_postprocess.pid")
        or int(reviewer_extra.get("complete", 0) or 0) >= 4
    )
    next_wave_runner_state = str(next_wave_runner.get("state", "")) if isinstance(next_wave_runner, dict) else ""

    rows = [
        status_row(
            "Comprehensive reviewer experiment plan exists",
            "docs/PUBLICATION_PLAN.md; docs/REVIEWER_EXPERIMENT_MATRIX.md; docs/REVIEWER_GOAL_COMPLETION_AUDIT.md",
            exists("docs/PUBLICATION_PLAN.md")
            and exists("docs/REVIEWER_EXPERIMENT_MATRIX.md")
            and exists("docs/REVIEWER_GOAL_COMPLETION_AUDIT.md"),
        ),
        status_row(
            "Faithful Drifting plan exists",
            "docs/DRIFTING_FAITHFULNESS_PLAN.md",
            exists("docs/DRIFTING_FAITHFULNESS_PLAN.md"),
        ),
        status_row(
            "Prompt-to-artifact checklist exists",
            "docs/REVIEWER_PROMPT_TO_ARTIFACT_CHECKLIST.md",
            exists("docs/REVIEWER_PROMPT_TO_ARTIFACT_CHECKLIST.md"),
        ),
        status_row(
            "Equation-to-code algorithm audit exists",
            "docs/DRIFTING_ALGORITHM_AUDIT.md",
            exists("docs/DRIFTING_ALGORITHM_AUDIT.md"),
        ),
        status_row(
            "Algorithm 2 tensor equivalence passes",
            "results/drifting_faithfulness_status.json",
            faith.get("algorithm2_equivalence", {}).get("status") == "PASS",
            f"max_abs_diff={faith.get('algorithm2_equivalence', {}).get('max_abs_diff')}",
        ),
        status_row(
            "Strict reviewer-faithful config protocol passes",
            "results/drifting_faithfulness_status.json",
            faith.get("strict_protocol_configs", {}).get("status") == "PASS",
            f"checked={faith.get('strict_protocol_configs', {}).get('checked')}, "
            f"failures={faith.get('strict_protocol_configs', {}).get('failures')}",
        ),
        status_row(
            "Reviewer-faithful manifest has strict core and allocation configs",
            "configs/reviewer_faithful/manifest.json",
            faithful_manifest["entries"] == 10
            and faithful_manifest["group_counts"].get("faithful_core") == 4
            and faithful_manifest["group_counts"].get("faithful_allocation") == 6
            and not faithful_manifest["missing_configs"],
            "entries={entries}, groups={groups}, missing_configs={missing}".format(
                entries=faithful_manifest["entries"],
                groups=faithful_manifest["group_counts"],
                missing=faithful_manifest["missing_configs"],
            ),
        ),
        status_row(
            "Faithful result collection artifacts exist",
            "results/faithful_drifting.csv; results/tables/tab_faithful_drifting_core.tex; docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex; DriftingMol_AAAI_FaithfulSupplement.pdf",
            exists("results/faithful_drifting.csv")
            and exists("results/tables/tab_faithful_drifting_core.tex")
            and exists("results/tables/tab_faithful_drifting_allocation.tex")
            and exists("scripts/render_faithful_supplement.py")
            and exists("docs/SUPPLEMENT_FAITHFUL_DRIFTING.tex")
            and exists("docs/SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex")
            and exists("docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex")
            and exists("DriftingMol_AAAI_FaithfulSupplement.pdf"),
        ),
        status_row(
            "Strict faithful core runs are complete",
            "results/faithful_drifting_status.json",
            faithful_core_complete,
            f"complete={faithful_runs.get('groups', {}).get('faithful_core', {}).get('complete', 0)}/"
            f"{faithful_runs.get('groups', {}).get('faithful_core', {}).get('total', 0)}",
        ),
        status_row(
            "Strict faithful allocation sweeps are complete",
            "results/faithful_drifting_status.json; results/tables/tab_faithful_drifting_allocation.tex",
            faithful_allocation_complete and exists("results/tables/tab_faithful_drifting_allocation.tex"),
            f"complete={faithful_runs.get('groups', {}).get('faithful_allocation', {}).get('complete', 0)}/"
            f"{faithful_runs.get('groups', {}).get('faithful_allocation', {}).get('total', 0)}",
        ),
        status_row(
            "Deferred faithful launcher completed or is active",
            "outputs/reviewer_faithful/deferred_faithful_core_launcher.pid; outputs/reviewer_faithful/core_status.json",
            deferred_launcher_ok,
            f"pid_alive={pid_alive('outputs/reviewer_faithful/deferred_faithful_core_launcher.pid')}, "
            f"core_state={core_runner.get('state')}, core_complete={faithful_core_complete}",
        ),
        status_row(
            "Destructive anti-symmetry evidence has minimum completed runs",
            "results/destructive_ablation_status.json",
            destructive.get("minimum_completed_runs_reached") is True,
            f"complete={destructive.get('complete', 0)}/{destructive.get('num_experiments', 0)}",
        ),
        status_row(
            "VAE sensitivity has at least one completed alternative",
            "results/vae_sensitivity_status.json",
            int(vae.get("complete", 0) or 0) >= 1,
            f"complete={vae.get('complete', 0)}/{vae.get('num_experiments', 0)}",
        ),
        status_row(
            "Graph archive launchability preflight is recorded",
            "scripts/audit_graph_archive_launchability.py; results/graph_archive_launchability_status.json; results/graph_archive_launchability_audit.md",
            exists("scripts/audit_graph_archive_launchability.py")
            and exists("results/graph_archive_launchability_status.json")
            and exists("results/graph_archive_launchability_audit.md"),
            "complete={complete}, missing_artifacts={missing}, blockers={blockers}, archived_metrics={metrics}, graph_cache={graph_cache}, legacy_latent_caches={legacy}".format(
                complete=graph_launchability.get("complete"),
                missing=len(graph_launchability.get("missing_required_artifacts", []) or []),
                blockers=len(graph_launchability.get("namespace", {}).get("blockers", []) or []),
                metrics=len([
                    row for row in graph_launchability.get("archived_diagnostic_runs", []) or []
                    if row.get("metrics_exists")
                ]),
                graph_cache=graph_launchability.get("recovery_candidates", {}).get("graph_cache", {}).get("exists"),
                legacy=len(graph_launchability.get("recovery_candidates", {}).get("legacy_latent_caches", []) or []),
            ),
        ),
        status_row(
            "Graph stress prepared manifest exists",
            "configs/publication_ext/graph_stress_manifest.json; docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md",
            exists("configs/publication_ext/graph_stress_manifest.json")
            and exists("docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md")
            and len(graph_stress_manifest.get("entries", []) or []) >= 5
            and len(graph_stress_manifest.get("preconditions", []) or []) >= 4,
            "entries={entries}, preconditions={preconditions}, launch_now={launch_now}, namespace_plan={plan}".format(
                entries=len(graph_stress_manifest.get("entries", []) or []),
                preconditions=len(graph_stress_manifest.get("preconditions", []) or []),
                launch_now=graph_stress_manifest.get("resource_policy", {}).get("launch_now"),
                plan=exists("docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md"),
            ),
        ),
        status_row(
            "Downstream VAE-drift queue and collector are organized",
            "configs/publication_ext/vae_drift_manifest.json; scripts/collect_vae_drift_results.py; results/vae_drift_downstream_status.json",
            vae_drift_manifest["entries"] == 4
            and vae_drift_manifest["group_counts"].get("vae_drift_downstream") == 4
            and not vae_drift_manifest["missing_configs"]
            and exists("scripts/collect_vae_drift_results.py")
            and exists("results/vae_drift_downstream_status.json")
            and exists("results/tables/tab_vae_drift_downstream.tex")
            and vae_drift_automation_present
            and vae_drift_postprocess_ok,
            "entries={entries}, complete={complete}/{total}, launchers_alive={alive}, automation_present={automation}, postprocess_ok={postprocess}, missing_configs={missing}".format(
                entries=vae_drift_manifest["entries"],
                complete=vae_drift.get("complete", 0),
                total=vae_drift.get("num_experiments", 0),
                alive=vae_drift_launchers_alive,
                automation=vae_drift_automation_present,
                postprocess=vae_drift_postprocess_ok,
                missing=vae_drift_manifest["missing_configs"],
            ),
        ),
        status_row(
            "Generalization queue and collector are organized",
            "configs/publication_ext/generalization_manifest.json; scripts/collect_generalization_results.py; results/generalization_status.json",
            generalization_manifest["entries"] == 4
            and generalization_manifest["group_counts"].get("single_property_generalization") == 2
            and generalization_manifest["group_counts"].get("multi4_seed_stability") == 2
            and not generalization_manifest["missing_configs"]
            and exists("scripts/collect_generalization_results.py")
            and exists("results/generalization_status.json")
            and exists("results/tables/tab_generalization.tex")
            and (generalization_launchers_alive >= 1 or int(generalization.get("complete", 0) or 0) >= 1)
            and generalization_postprocess_ok,
            "entries={entries}, complete={complete}/{total}, launchers_alive={alive}, postprocess_ok={postprocess}, missing_configs={missing}".format(
                entries=generalization_manifest["entries"],
                complete=generalization.get("complete", 0),
                total=generalization.get("num_experiments", 0),
                alive=generalization_launchers_alive,
                postprocess=generalization_postprocess_ok,
                missing=generalization_manifest["missing_configs"],
            ),
        ),
        status_row(
            "Reviewer-extra queue and collector are organized",
            "configs/publication_ext/reviewer_extra_manifest.json; scripts/collect_reviewer_extra_results.py; results/reviewer_extra_status.json",
            reviewer_extra_manifest["entries"] == 4
            and reviewer_extra_manifest["group_counts"].get("continuous_conditioning") == 1
            and reviewer_extra_manifest["group_counts"].get("single_property_seed_extension") == 2
            and reviewer_extra_manifest["group_counts"].get("vae_drift_seed_extension") == 1
            and not reviewer_extra_manifest["missing_configs"]
            and exists("scripts/collect_reviewer_extra_results.py")
            and exists("scripts/watch_reviewer_extra_postprocess.py")
            and exists("results/reviewer_extra_status.json")
            and exists("results/tables/tab_reviewer_extra.tex")
            and (reviewer_extra_launchers_alive >= 1 or int(reviewer_extra.get("complete", 0) or 0) >= 1)
            and reviewer_extra_postprocess_ok,
            "entries={entries}, complete={complete}/{total}, launchers_alive={alive}, postprocess_ok={postprocess}, missing_configs={missing}".format(
                entries=reviewer_extra_manifest["entries"],
                complete=reviewer_extra.get("complete", 0),
                total=reviewer_extra.get("num_experiments", 0),
                alive=reviewer_extra_launchers_alive,
                postprocess=reviewer_extra_postprocess_ok,
                missing=reviewer_extra_manifest["missing_configs"],
            ),
        ),
        status_row(
            "Next-wave reviewer experiments are prepared or running",
            "docs/NEXT_WAVE_EXPERIMENT_PLAN.md; configs/publication_ext/next_wave_manifest.json; scripts/collect_next_wave_results.py; results/next_wave_status.json; outputs/publication_ext/parallel_runner_status_next_wave_gpu0.json",
            next_wave_manifest["entries"] == 4
            and next_wave_manifest["group_counts"].get("property_guidance_baseline") == 3
            and next_wave_manifest["group_counts"].get("conditioning_seed_stability") == 1
            and not next_wave_manifest["missing_configs"]
            and exists("docs/NEXT_WAVE_EXPERIMENT_PLAN.md")
            and exists("scripts/generate_next_wave_configs.py")
            and exists("scripts/collect_next_wave_results.py")
            and exists("results/next_wave_status.json")
            and exists("results/tables/tab_next_wave.tex"),
            "entries={entries}, complete={complete}/{total}, runner_state={runner_state}, missing_configs={missing}".format(
                entries=next_wave_manifest["entries"],
                complete=next_wave.get("complete", 0),
                total=next_wave.get("num_experiments", 0),
                runner_state=next_wave_runner_state or "none",
                missing=next_wave_manifest["missing_configs"],
            ),
        ),
        status_row(
            "Trained baseline queue and collector are organized",
            "configs/publication_ext/baseline_manifest.json; scripts/collect_trained_baselines.py; results/trained_baseline_status.json",
            baseline_manifest["entries"] == 3
            and baseline_manifest["group_counts"].get("trained_baseline") == 3
            and not baseline_manifest["missing_configs"]
            and exists("scripts/collect_trained_baselines.py")
            and exists("results/trained_baseline_status.json")
            and exists("results/tables/tab_trained_baseline_qed.tex"),
            "entries={entries}, complete={complete}/{total}, missing_configs={missing}".format(
                entries=baseline_manifest["entries"],
                complete=trained_baselines.get("complete", 0),
                total=trained_baselines.get("num_experiments", 0),
                missing=baseline_manifest["missing_configs"],
            ),
        ),
        status_row(
            "Overall extension audit is closed",
            "results/extension_completion_status.json",
            bool(ext.get("complete", False)),
            f"complete={ext.get('complete', False)}",
        ),
    ]
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "OPEN"
    return {"overall": overall, "rows": rows}


def write_report(status: dict[str, Any]) -> None:
    lines = [
        "# Reviewer Experiment Readiness Audit",
        "",
        "Objective: verify the planning and execution state for reviewer-facing",
        "experiments, especially faithful reproduction of Drifting Models.",
        "",
        "| Requirement | Evidence | Status | Note |",
        "|---|---|---|---|",
    ]
    for row in status["rows"]:
        lines.append(
            f"| {row['requirement']} | `{row['evidence']}` | {row['status']} | {row.get('note', '')} |"
        )
    lines += ["", f"Overall: {status['overall']}", ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    status = build_status()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(status, indent=2) + "\n")
    write_report(status)
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Overall reviewer experiment readiness: {status['overall']}")
    return 0 if status["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
