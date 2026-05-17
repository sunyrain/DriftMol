#!/usr/bin/env python3
"""Generate next-wave reviewer experiments without launching them.

The active extension queue already occupies all GPUs.  These configs are the
highest-value follow-up jobs to run only after the current VAE-drift,
generalization, and reviewer-extra queues produce final metrics.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "publication_ext" / "next_wave"
MANIFEST = ROOT / "configs" / "publication_ext" / "next_wave_manifest.json"


def load_yaml(rel_path: str) -> dict[str, Any]:
    with (ROOT / rel_path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {rel_path}")
    return data


def dump_yaml(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def deep_set(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def clone_config(
    base: dict[str, Any],
    name: str,
    output_dir: str,
    seed: int,
    updates: dict[tuple[str, ...], Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    deep_set(cfg, ("experiment", "name"), name)
    deep_set(cfg, ("experiment", "output_dir"), output_dir)
    deep_set(cfg, ("experiment", "seed"), seed)
    for key_path, value in (updates or {}).items():
        deep_set(cfg, key_path, value)
    return cfg


def command_for(config_path: Path) -> str:
    return f"python -m src.train.train_selfies_cfg --config {config_path.relative_to(ROOT)}"


def add_entry(
    entries: list[dict[str, Any]],
    group: str,
    target_property: str,
    cfg: dict[str, Any],
    display: str,
    purpose: str,
    recommended_wait_for: list[str],
    comparator: str,
) -> None:
    path = OUT_DIR / f"{cfg['experiment']['name']}.yaml"
    dump_yaml(cfg, path)
    entries.append(
        {
            "group": group,
            "name": cfg["experiment"]["name"],
            "display": display,
            "target_property": target_property,
            "config": str(path.relative_to(ROOT)),
            "output_dir": cfg["experiment"]["output_dir"],
            "command": command_for(path),
            "purpose": purpose,
            "recommended_wait_for": recommended_wait_for,
            "comparator": comparator,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    base_linear = load_yaml("configs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s42.yaml")
    base_qed = load_yaml("configs/final/exp_G4_qed.yaml")

    linear_specs = [
        (
            "ext_NW_LINEAR_PROP_LOGP_s42",
            "LogP linear property-guidance baseline",
            "logp",
            [2],
            1,
            "tests whether LogP transfer is explainable by a fixed linear latent-property head",
            ["outputs/publication_ext/generalization/ext_G4_logp_qed_s42/final_metrics.json"],
            "ext_G4_logp_qed_s42 and ext_G4_logp_qed_s43",
        ),
        (
            "ext_NW_LINEAR_PROP_SA_s42",
            "SA linear property-guidance baseline",
            "sa_score",
            [1],
            1,
            "tests whether SA-score transfer is explainable by a fixed linear latent-property head",
            ["outputs/publication_ext/generalization/ext_G4_sa_qed_s42/final_metrics.json"],
            "ext_G4_sa_qed_s42 and ext_G4_sa_qed_s43",
        ),
        (
            "ext_NW_LINEAR_PROP_MULTI4_s42",
            "Multi-property linear guidance baseline",
            "multi4",
            [0, 1, 2, 3],
            4,
            "tests whether four-property no-binning control needs Drifting rather than fixed linear guidance",
            [
                "outputs/publication_ext/generalization/ext_G4_multi4_v2_s43/final_metrics.json",
                "outputs/publication_ext/generalization/ext_G4_multi4_v2_s44/final_metrics.json",
            ],
            "pub_G4_multi4_v2_s42 plus ext_G4_multi4_v2_s43/s44",
        ),
    ]
    for name, display, target, prop_indices, cond_dim, purpose, wait_for, comparator in linear_specs:
        cfg = clone_config(
            base_linear,
            name=name,
            output_dir=f"outputs/publication_ext/next_wave/{name}",
            seed=42,
            updates={
                ("data", "prop_indices"): prop_indices,
                ("generator", "cond_dim"): cond_dim,
                ("cond_binning", "enabled"): False,
                ("loss", "lambda_drift"): 0.0,
                ("loss", "lambda_prop"): 1.0,
                ("prop_head", "type"): "linear",
                ("prop_head", "ridge_alpha"): 10.0,
            },
        )
        add_entry(
            entries,
            group="property_guidance_baseline",
            target_property=target,
            cfg=cfg,
            display=display,
            purpose=purpose,
            recommended_wait_for=wait_for,
            comparator=comparator,
        )

    cont_qed = clone_config(
        base_qed,
        name="ext_NW_QED_CONTINUOUS_s43",
        output_dir="outputs/publication_ext/next_wave/ext_NW_QED_CONTINUOUS_s43",
        seed=43,
        updates={
            ("cond_binning", "enabled"): False,
            ("cfg", "positive_mode"): "prop",
        },
    )
    add_entry(
        entries,
        group="conditioning_seed_stability",
        target_property="qed",
        cfg=cont_qed,
        display="Continuous QED control, seed 43",
        purpose="adds a second seed for the no-binning QED control test",
        recommended_wait_for=[
            "outputs/publication_ext/reviewer_extra/ext_G4_qed_continuous_s42/final_metrics.json"
        ],
        comparator="ext_G4_qed_continuous_s42",
    )

    payload = {
        "description": "Next-wave reviewer experiments prepared but not launched while GPUs and disk are constrained.",
        "resource_policy": {
            "launch_now": False,
            "reason": "All four GPUs are occupied by active VAE-drift jobs and /root/autodl-tmp has limited free disk.",
            "minimum_recommended_disk_free_gb": 20,
        },
        "notes": [
            "These jobs are intentionally outside the strict extension completion gate.",
            "Launch after the current VAE-drift, generalization, and reviewer-extra queues finish or after explicit re-prioritization.",
            "The first three rows strengthen baseline coverage for property-transfer claims.",
            "The fourth row strengthens the continuous-conditioning claim if the seed-42 run is promising.",
        ],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(entries)} next-wave configs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
