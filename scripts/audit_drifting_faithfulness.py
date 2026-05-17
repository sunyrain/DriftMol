#!/usr/bin/env python3
"""Audit evidence for faithful reproduction of the Drifting Models recipe."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drifting.drift_latent_phi import compute_drift_field_paper

MANIFEST = ROOT / "configs" / "reviewer_faithful" / "manifest.json"
OUT_JSON = ROOT / "results" / "drifting_faithfulness_status.json"
OUT_MD = ROOT / "results" / "drifting_faithfulness_audit.md"


def reference_algorithm2(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Direct transcription of Drifting Models Algorithm 2."""
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, x)
    dist_neg = dist_neg + torch.eye(x.shape[0]) * 1e6
    logit_pos = -dist_pos / temperature
    logit_neg = -dist_neg / temperature
    logit = torch.cat([logit_pos, logit_neg], dim=1)
    a_row = logit.softmax(dim=-1)
    a_col = logit.softmax(dim=-2)
    a = (a_row * a_col).sqrt()
    a_pos = a[:, : y_pos.shape[0]]
    a_neg = a[:, y_pos.shape[0] :]
    w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)
    return w_pos @ y_pos - w_neg @ x


def algorithm2_equivalence() -> dict[str, Any]:
    torch.manual_seed(7)
    x = torch.randn(9, 13)
    y_pos = torch.randn(11, 13)
    temperature = 0.05
    v_ref = reference_algorithm2(x, y_pos, temperature)
    v_impl = compute_drift_field_paper(
        x,
        y_pos,
        temperature=temperature,
        normalize_distances=False,
        norm_mode="xy",
        attraction_scale=1.0,
        repulsion_scale=1.0,
    )
    max_abs_diff = float((v_ref - v_impl).abs().max().item())
    return {
        "status": "PASS" if max_abs_diff < 1e-6 else "FAIL",
        "max_abs_diff": max_abs_diff,
        "temperature": temperature,
        "shape": list(x.shape),
        "n_pos": int(y_pos.shape[0]),
    }


def manifest_status() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {"status": "OPEN", "reason": "manifest missing", "entries": 0}
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest.get("entries", [])
    missing_configs = [entry["config"] for entry in entries if not (ROOT / entry["config"]).exists()]
    groups: dict[str, int] = {}
    completed = 0
    for entry in entries:
        groups[entry.get("group", "")] = groups.get(entry.get("group", ""), 0) + 1
        if (ROOT / entry["output_dir"] / "final_metrics.json").exists():
            completed += 1
    return {
        "status": "PASS" if entries and not missing_configs else "OPEN",
        "entries": len(entries),
        "groups": groups,
        "missing_configs": missing_configs,
        "completed_runs": completed,
        "pending_runs": len(entries) - completed,
    }


def _get(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _is_zero(value: Any) -> bool:
    try:
        return abs(float(value)) < 1e-12
    except (TypeError, ValueError):
        return False


def strict_protocol_config_status() -> dict[str, Any]:
    """Verify reviewer-faithful configs encode the intended protocol."""
    if not MANIFEST.exists():
        return {"status": "OPEN", "reason": "manifest missing", "checked": 0}
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest.get("entries", [])
    failures: list[str] = []
    checked = 0

    for entry in entries:
        if not isinstance(entry, dict):
            failures.append("malformed manifest entry")
            continue
        name = entry.get("name", "<unnamed>")
        group = entry.get("group", "")
        config = entry.get("config")
        if not isinstance(config, str) or not (ROOT / config).exists():
            failures.append(f"{name}: config missing")
            continue
        cfg = yaml.safe_load((ROOT / config).read_text())
        checked += 1

        common_expectations = [
            (_get(cfg, "training", "epochs") == 100, "training.epochs != 100"),
            (_get(cfg, "loss", "lambda_decoupled_drift") == 0.0, "decoder-coupled drift enabled"),
            (_get(cfg, "loss", "lambda_dec_drift") == 0.0, "decoder drift enabled"),
            (_is_zero(_get(cfg, "loss", "lambda_zdiv")), "z-diversity enabled"),
            (_is_zero(_get(cfg, "loss", "lambda_phidiv")), "phi-diversity enabled"),
            (_get(cfg, "loss", "temperatures") == [0.02, 0.05, 0.2], "temperature set mismatch"),
            (_get(cfg, "loss", "drift_normalize") is True, "kernel normalization disabled"),
            (
                _get(cfg, "loss", "drift_normalize_dist") is True,
                "Paper A.6 feature-distance normalization disabled",
            ),
            (_get(cfg, "loss", "drift_norm_mode") == "xy", "bidirectional normalization mode mismatch"),
            (_get(cfg, "loss", "drift_attraction_scale") == 1.0, "attraction scale changed"),
            (_get(cfg, "loss", "drift_repulsion_scale") == 1.0, "repulsion scale changed"),
            (_get(cfg, "cfg", "positive_mode") == "prop", "positive sampling mode mismatch"),
            (_get(cfg, "cfg", "alpha_power") == 3, "alpha sampling power mismatch"),
            (_get(cfg, "cfg", "alpha_min") == 1.0, "alpha_min mismatch"),
            (_get(cfg, "cfg", "alpha_max") == 4.0, "alpha_max mismatch"),
            (_get(cfg, "cond_binning", "enabled") is True, "QED binning disabled"),
            (_get(cfg, "cond_binning", "method") == "quantile", "QED binning is not quantile"),
        ]
        for ok, reason in common_expectations:
            if not ok:
                failures.append(f"{name}: {reason}")

        if group == "faithful_core":
            core_expectations = [
                (_get(cfg, "cfg", "n_groups") == 64, "N_c != 64"),
                (_get(cfg, "cfg", "n_gen") == 64, "N_neg != 64"),
                (_get(cfg, "cfg", "n_pos") == 64, "N_pos != 64"),
            ]
            for ok, reason in core_expectations:
                if not ok:
                    failures.append(f"{name}: {reason}")

        if name == "rf_FD_STRICT_PLAIN_PHI_QED_s42":
            if _get(cfg, "feature_space", "mode") != "phi":
                failures.append(f"{name}: feature_space.mode is not phi")
            if "zinc_phi_plain" not in str(_get(cfg, "phi", "checkpoint", default="")):
                failures.append(f"{name}: plain latent-MAE checkpoint missing")
        elif name == "rf_FD_STRICT_PROP_PHI_QED_s42":
            if _get(cfg, "feature_space", "mode") != "phi":
                failures.append(f"{name}: feature_space.mode is not phi")
            if "zinc_phi_prop" not in str(_get(cfg, "phi", "checkpoint", default="")):
                failures.append(f"{name}: property-aware latent-MAE checkpoint missing")
        elif name == "rf_FD_STRICT_RANDOM_PHI_QED_s42":
            if _get(cfg, "feature_space", "mode") != "random":
                failures.append(f"{name}: feature_space.mode is not random")
        elif name == "rf_FD_STRICT_ZSPACE_QED_s42":
            if _get(cfg, "loss", "lambda_drift") != 0.0 or _get(cfg, "loss", "lambda_zdrift") != 1.0:
                failures.append(f"{name}: z-space control drift weights mismatch")
        elif group == "faithful_core":
            failures.append(f"{name}: unexpected faithful_core run")

    return {
        "status": "PASS" if checked == 10 and not failures else "OPEN",
        "checked": checked,
        "failures": failures,
    }


def strict_run_completion_status(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest is None:
        manifest = manifest_status()
    entries = int(manifest.get("entries", 0))
    completed = int(manifest.get("completed_runs", 0))
    pending = int(manifest.get("pending_runs", max(entries - completed, 0)))
    return {
        "status": "PASS" if entries > 0 and pending == 0 else "OPEN",
        "completed_runs": completed,
        "pending_runs": pending,
        "required_runs": entries,
    }


def phi_checkpoint_status() -> dict[str, Any]:
    paths = {
        "plain_latent_mae": ROOT / "outputs" / "foundation" / "zinc_phi_plain" / "best_latent_mae.pt",
        "property_latent_mae": ROOT / "outputs" / "foundation" / "zinc_phi_prop" / "best_latent_mae.pt",
    }
    exists = {name: path.exists() for name, path in paths.items()}
    return {
        "status": "PASS" if all(exists.values()) else "OPEN",
        "checkpoints": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        "exists": exists,
    }


def existing_phi_result_status() -> dict[str, Any]:
    csv_path = ROOT / "results" / "publication_results.csv"
    wanted = {"C1", "C2", "C3", "C4", "C5"}
    found: dict[str, dict[str, str]] = {}
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if (
                    row.get("condition") == "qed"
                    and row.get("variant") in wanted
                    and row.get("status", "").startswith("complete")
                ):
                    found[row["variant"]] = {
                        "status": row["status"],
                        "rho": row.get("spearman_rho", ""),
                        "uniqueness": row.get("uniqueness", ""),
                        "root": row.get("root", ""),
                    }
    return {
        "status": "PASS" if wanted.issubset(found.keys()) else "OPEN",
        "found": found,
        "missing": sorted(wanted.difference(found.keys())),
    }


def collector_artifact_status() -> dict[str, Any]:
    paths = [
        ROOT / "scripts" / "collect_faithful_drifting_results.py",
        ROOT / "scripts" / "render_faithful_supplement.py",
        ROOT / "scripts" / "defer_faithful_core_after_destructive.py",
        ROOT / "docs" / "DRIFTING_ALGORITHM_AUDIT.md",
        ROOT / "docs" / "SUPPLEMENT_FAITHFUL_DRIFTING.tex",
        ROOT / "docs" / "SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex",
        ROOT / "docs" / "SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex",
        ROOT / "DriftingMol_AAAI_FaithfulSupplement.pdf",
        ROOT / "results" / "faithful_drifting.csv",
        ROOT / "results" / "faithful_drifting_status.json",
        ROOT / "results" / "tables" / "tab_faithful_drifting_core.tex",
        ROOT / "results" / "tables" / "tab_faithful_drifting_allocation.tex",
        ROOT / "docs" / "DRIFTING_FAITHFULNESS_EXECUTION_CHECKLIST.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    return {
        "status": "PASS" if not missing else "OPEN",
        "missing": missing,
        "artifacts": [str(path.relative_to(ROOT)) for path in paths],
    }


def destructive_status() -> dict[str, Any]:
    path = ROOT / "results" / "destructive_ablation_status.json"
    if not path.exists():
        return {"status": "OPEN", "reason": "destructive status missing"}
    payload = json.loads(path.read_text())
    return {
        "status": "PASS" if payload.get("minimum_completed_runs_reached") else "OPEN",
        "complete": payload.get("complete", 0),
        "pending_or_incomplete": payload.get("pending_or_incomplete", 0),
        "status_counts": payload.get("status_counts", {}),
    }


def write_report(status: dict[str, Any]) -> None:
    rows = [
        ("Algorithm 2 implementation matches direct transcription", "algorithm2_equivalence"),
        ("Reviewer-faithful manifest and configs exist", "manifest"),
        ("Reviewer-faithful configs match the strict Drifting protocol", "strict_protocol_configs"),
        ("Strict reviewer-faithful runs are complete", "strict_run_completion"),
        ("Faithfulness result collection artifacts exist", "collector_artifacts"),
        ("Frozen latent-MAE feature extractors exist", "phi_checkpoints"),
        ("Existing C1-C5 phi-space results are collected", "existing_phi_results"),
        ("Destructive anti-symmetry ablations have minimum completed evidence", "destructive_ablations"),
    ]
    lines = [
        "# Drifting Faithfulness Audit",
        "",
        "Objective: verify whether the repository contains concrete evidence and",
        "runnable experiments for faithful reproduction of the original Drifting",
        "Models algorithm before molecule-specific decoder-coupled modifications.",
        "",
        "| Requirement | Evidence key | Status |",
        "|---|---|---|",
    ]
    for label, key in rows:
        lines.append(f"| {label} | `{key}` | {status[key]['status']} |")
    if status["strict_run_completion"]["status"] == "PASS":
        completion_note = [
            "Strict reviewer-faithful runs are intentionally reported separately from",
            "the already completed C1-C5 phi-space experiments. The reviewer-faithful",
            "manifest is now complete, so these rows are final supplemental evidence",
            "for the faithful reproduction package.",
        ]
    else:
        completion_note = [
            "Strict reviewer-faithful runs are intentionally reported separately from",
            "the already completed C1-C5 phi-space experiments. They should remain",
            "queued evidence until the reviewer-faithful manifest has no pending runs.",
        ]
    lines += [
        "",
        f"Overall: {status['overall']}",
        "",
        *completion_note,
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    manifest = manifest_status()
    status = {
        "algorithm2_equivalence": algorithm2_equivalence(),
        "manifest": manifest,
        "strict_protocol_configs": strict_protocol_config_status(),
        "strict_run_completion": strict_run_completion_status(manifest),
        "collector_artifacts": collector_artifact_status(),
        "phi_checkpoints": phi_checkpoint_status(),
        "existing_phi_results": existing_phi_result_status(),
        "destructive_ablations": destructive_status(),
    }
    status["overall"] = (
        "PASS"
        if all(item["status"] == "PASS" for item in status.values())
        else "OPEN"
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(status, indent=2) + "\n")
    write_report(status)
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Overall Drifting faithfulness audit: {status['overall']}")
    return 0 if status["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
