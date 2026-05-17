#!/usr/bin/env python3
"""Audit whether the archived graph route can be launched safely.

This is a preflight for the graph-stress follow-up, not a result collector.
The archived graph line still uses the old ``src.*`` import namespace while the
current repository uses ``src`` for the SELFIES line. The audit records whether
fresh graph-control runs are launchable without accidentally importing the
wrong modules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "graph_vae_line"
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "graph_archive_launchability_status.json"
OUT_MD = RESULTS / "graph_archive_launchability_audit.md"

GRAPH_CONFIGS = [
    ARCHIVE / "configs" / "e36_dec_drift_cfg.yaml",
    ARCHIVE / "configs" / "e40_logp_bins_queue.yaml",
]
REQUIRED_CONFIG_PATHS = [
    ("vae", "checkpoint"),
    ("phi", "checkpoint"),
    ("data", "latent_cache_path"),
]
GRAPH_DIAGNOSTIC_RUNS = [
    ("e36_dec_drift_cfg", "qed"),
    ("e40_logp_bins_queue", "logp"),
]
SRC_IMPORT_RE = re.compile(r"(?:from|import)\s+(src(?:\.[A-Za-z0-9_]+)+)")


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required for graph launchability audit") from exc

    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text())
    return payload if isinstance(payload, dict) else {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def nested_get(payload: dict[str, Any], keys: tuple[str, str]) -> str:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
    return cur if isinstance(cur, str) else ""


def config_artifact_checks(archive: Path = ARCHIVE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg_path in GRAPH_CONFIGS:
        cfg_path = archive / cfg_path.relative_to(ARCHIVE)
        payload = load_yaml(cfg_path)
        for keys in REQUIRED_CONFIG_PATHS:
            configured = nested_get(payload, keys)
            resolved = archive / configured if configured else archive
            rows.append(
                {
                    "config": rel(cfg_path),
                    "field": ".".join(keys),
                    "configured_path": configured,
                    "resolved_path": rel(resolved),
                    "exists": bool(configured) and resolved.exists(),
                }
            )
    return rows


def scan_src_imports(archive: Path = ARCHIVE) -> list[str]:
    imports: set[str] = set()
    for path in archive.rglob("*.py"):
        text = path.read_text(errors="replace")
        imports.update(match.group(1) for match in SRC_IMPORT_RE.finditer(text))
    return sorted(imports)


def current_utils_graph_ready(root: Path = ROOT) -> dict[str, Any]:
    utils = root / "src" / "utils.py"
    text = utils.read_text(errors="replace") if utils.exists() else ""
    required = ["def load_vae", "def discretize_logits"]
    missing = [token for token in required if token not in text]
    return {
        "path": rel(utils, root),
        "exists": utils.exists(),
        "missing_graph_functions": missing,
        "ready": utils.exists() and not missing,
    }


def recovery_candidate_status(root: Path = ROOT) -> dict[str, Any]:
    graph_cache = root / "archive" / "data_qm9" / "qm9_graph_cache.pt"
    v2_latent = sorted((root / "archive" / "data_qm9").glob("qm9_latent_cache_v2*.pt"))
    raw_sdf = list(root.rglob("gdb9.sdf"))
    graph_root = root / "archive" / "graph_vae_line"
    vae_final = graph_root / "outputs" / "vae_v3_valence" / "final_metrics.json"
    latent_mae_log = graph_root / "outputs" / "latent_mae_v3_train.log"
    latent_mae_text = latent_mae_log.read_text(errors="replace") if latent_mae_log.exists() else ""
    return {
        "graph_cache": {
            "path": rel(graph_cache, root),
            "exists": graph_cache.exists(),
        },
        "legacy_latent_caches": [rel(path, root) for path in v2_latent],
        "raw_sdf_candidates": [rel(path, root) for path in raw_sdf],
        "vae_v3_valence_final_metrics": {
            "path": rel(vae_final, root),
            "exists": vae_final.exists(),
        },
        "latent_mae_v3_train_log": {
            "path": rel(latent_mae_log, root),
            "exists": latent_mae_log.exists(),
            "mentions_checkpoint": "best_latent_mae.pt" in latent_mae_text,
        },
    }


def _best_prop_control(metrics: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for key, value in metrics.items():
        if not key.startswith("prop_control_a") or not isinstance(value, dict):
            continue
        rho = value.get("spearman_rho")
        if not isinstance(rho, (int, float)):
            continue
        if not best or float(rho) > float(best.get("spearman_rho", float("-inf"))):
            alpha = key.replace("prop_control_a", "")
            best = {
                "alpha": alpha,
                "spearman_rho": float(rho),
                "prop_gap": value.get("prop_gap"),
                "n_valid_corr": value.get("n_valid_corr"),
            }
    return best


def _generation_at_alpha(metrics: dict[str, Any], alpha: str) -> dict[str, Any]:
    if not alpha:
        return {}
    generation = metrics.get(f"generation_cfg_a{alpha}", {})
    return generation if isinstance(generation, dict) else {}


def archived_diagnostic_status(root: Path = ROOT, archive: Path = ARCHIVE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, prop in GRAPH_DIAGNOSTIC_RUNS:
        out_dir = archive / "outputs" / name
        metrics_path = out_dir / "final_metrics.json"
        resolved_path = out_dir / "resolved_config.json"
        metrics = load_json(metrics_path)
        best = _best_prop_control(metrics)
        generation = _generation_at_alpha(metrics, str(best.get("alpha", "")))
        rows.append(
            {
                "name": name,
                "property": prop,
                "metrics_path": rel(metrics_path, root),
                "resolved_config_path": rel(resolved_path, root),
                "metrics_exists": metrics_path.exists(),
                "resolved_config_exists": resolved_path.exists(),
                "best_prop_control": best,
                "generation_at_best_alpha": {
                    "validity": generation.get("validity"),
                    "uniqueness": generation.get("uniqueness"),
                    "novelty": generation.get("novelty"),
                },
            }
        )
    return rows


def build_status(root: Path = ROOT, archive: Path = ARCHIVE) -> dict[str, Any]:
    artifacts = config_artifact_checks(archive)
    imports = scan_src_imports(archive)
    archive_src = archive / "src"
    archive_src_utils = archive_src / "utils.py"
    namespace_plan = root / "docs" / "GRAPH_NAMESPACE_ADAPTER_PLAN.md"
    current_utils = current_utils_graph_ready(root)
    missing_artifacts = [row for row in artifacts if not row["exists"]]

    namespace_blockers: list[str] = []
    if imports and not archive_src.exists():
        namespace_blockers.append("archived graph files import src.*, but archive/graph_vae_line/src is absent")
    if imports and not archive_src_utils.exists() and not current_utils["ready"]:
        namespace_blockers.append(
            "no graph-compatible src.utils is available; current src/utils.py lacks load_vae/discretize_logits"
        )

    complete = not missing_artifacts and not namespace_blockers
    return {
        "complete": complete,
        "archive_root": rel(archive, root),
        "config_artifacts": artifacts,
        "missing_required_artifacts": missing_artifacts,
        "namespace": {
            "src_imports": imports,
            "archive_src_package_exists": archive_src.exists(),
            "archive_src_utils_exists": archive_src_utils.exists(),
            "namespace_adapter_plan_exists": namespace_plan.exists(),
            "current_utils": current_utils,
            "blockers": namespace_blockers,
        },
        "archived_diagnostic_runs": archived_diagnostic_status(root, archive),
        "recovery_candidates": recovery_candidate_status(root),
    }


def write_markdown(status: dict[str, Any], path: Path = OUT_MD) -> None:
    rows = status["config_artifacts"]
    blockers = status["namespace"]["blockers"]
    lines = [
        "# Graph Archive Launchability Audit",
        "",
        f"Overall: {'PASS' if status['complete'] else 'OPEN'}",
        "",
        "## Required Artifacts",
        "",
        "| Config | Field | Path | Exists |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {field} | `{path}` | {exists} |".format(
                config=row["config"],
                field=row["field"],
                path=row["configured_path"] or "<missing>",
                exists="yes" if row["exists"] else "no",
            )
        )
    lines.extend(["", "## Namespace", ""])
    lines.append(
        f"- archived `src` package exists: {status['namespace']['archive_src_package_exists']}"
    )
    lines.append(
        f"- archived `src.utils` exists: {status['namespace']['archive_src_utils_exists']}"
    )
    lines.append(
        "- current `src.utils` graph-ready: "
        f"{status['namespace']['current_utils']['ready']}"
    )
    lines.append(
        "- namespace adapter plan exists: "
        f"{status['namespace'].get('namespace_adapter_plan_exists')}"
    )
    if blockers:
        lines.extend(["", "Blockers:"])
        lines.extend(f"- {blocker}" for blocker in blockers)
    diagnostics = status.get("archived_diagnostic_runs", [])
    if diagnostics:
        lines.extend(["", "## Archived Diagnostic Metrics", ""])
        lines.extend([
            "| Run | Property | Metrics | Best alpha | Best rho | Validity | Uniqueness | Novelty |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in diagnostics:
            best = row.get("best_prop_control", {})
            gen = row.get("generation_at_best_alpha", {})
            lines.append(
                "| {name} | {prop} | {metrics} | {alpha} | {rho} | {validity} | {uniq} | {novelty} |".format(
                    name=row["name"],
                    prop=row["property"],
                    metrics="yes" if row["metrics_exists"] else "no",
                    alpha=best.get("alpha", "-"),
                    rho=_fmt_float(best.get("spearman_rho")),
                    validity=_fmt_float(gen.get("validity")),
                    uniq=_fmt_float(gen.get("uniqueness")),
                    novelty=_fmt_float(gen.get("novelty")),
                )
            )
    recovery = status.get("recovery_candidates", {})
    if recovery:
        lines.extend(["", "## Recovery Candidates", ""])
        graph_cache = recovery.get("graph_cache", {})
        lines.append(
            f"- graph cache: `{graph_cache.get('path')}` exists={graph_cache.get('exists')}"
        )
        legacy = recovery.get("legacy_latent_caches", [])
        lines.append(f"- legacy latent caches: {len(legacy)}")
        lines.append(f"- raw gdb9.sdf candidates: {len(recovery.get('raw_sdf_candidates', []))}")
        vae_final = recovery.get("vae_v3_valence_final_metrics", {})
        lines.append(
            f"- VAE v3 final metrics: `{vae_final.get('path')}` exists={vae_final.get('exists')}"
        )
        mae_log = recovery.get("latent_mae_v3_train_log", {})
        lines.append(
            "- latent-MAE v3 train log: "
            f"`{mae_log.get('path')}` exists={mae_log.get('exists')}, "
            f"mentions checkpoint={mae_log.get('mentions_checkpoint')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "-"


def main() -> int:
    status = build_status()
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(status, indent=2) + "\n")
    write_markdown(status)
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    print(f"Overall graph archive launchability: {'PASS' if status['complete'] else 'OPEN'}")
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
