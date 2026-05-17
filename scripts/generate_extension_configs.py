#!/usr/bin/env python3
"""Generate extension-stage experiment configs and a runnable manifest.

These configs are intentionally separated from configs/publication/ so the
audited 8-page submission package remains stable while extension experiments
are added for the next revision cycle.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "publication_ext"
DESTRUCTIVE_DIR = OUT_DIR / "destructive"
MANIFEST = OUT_DIR / "manifest.json"


def load_yaml(rel_path: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / rel_path).read_text())


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
    seed: int = 42,
    updates: dict[tuple[str, ...], Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    deep_set(cfg, ("experiment", "name"), name)
    deep_set(cfg, ("experiment", "output_dir"), output_dir)
    deep_set(cfg, ("experiment", "seed"), seed)
    for key_path, value in (updates or {}).items():
        deep_set(cfg, key_path, value)
    return cfg


def command_for(config_path: Path, module: str = "src.train.train_selfies_cfg") -> str:
    return f"python -m {module} --config {config_path.relative_to(ROOT)}"


def add_entry(
    entries: list[dict[str, Any]],
    group: str,
    cfg: dict[str, Any],
    path: Path,
    purpose: str,
    module: str = "src.train.train_selfies_cfg",
) -> None:
    dump_yaml(cfg, path)
    entries.append(
        {
            "group": group,
            "name": cfg["experiment"]["name"],
            "config": str(path.relative_to(ROOT)),
            "output_dir": cfg["experiment"]["output_dir"],
            "command": command_for(path, module=module),
            "purpose": purpose,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DESTRUCTIVE_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    base_g4 = load_yaml("configs/final/exp_G4_qed.yaml")
    base_vae = load_yaml("configs/foundation/zinc_selfies_vae_v2.yaml")
    destructive_specs = [
        (
            "D_ATTR",
            "attraction-only drift; removes repulsive term V-",
            {
                ("loss", "drift_attraction_scale"): 1.0,
                ("loss", "drift_repulsion_scale"): 0.0,
            },
        ),
        (
            "D_REPL",
            "repulsion-only drift; removes attractive term V+",
            {
                ("loss", "drift_attraction_scale"): 0.0,
                ("loss", "drift_repulsion_scale"): 1.0,
            },
        ),
        (
            "D_BROKEN_ATTR",
            "broken anti-symmetry with overweighted attraction",
            {
                ("loss", "drift_attraction_scale"): 1.5,
                ("loss", "drift_repulsion_scale"): 1.0,
            },
        ),
        (
            "D_BROKEN_REPL",
            "broken anti-symmetry with overweighted repulsion",
            {
                ("loss", "drift_attraction_scale"): 1.0,
                ("loss", "drift_repulsion_scale"): 1.5,
            },
        ),
        (
            "D_YONLY",
            "remove x-axis component of bidimensional normalization",
            {
                ("loss", "drift_norm_mode"): "y",
            },
        ),
        (
            "D_NOCROSS",
            "remove cross-multiplication while keeping bidimensional softmax",
            {
                ("loss", "drift_norm_mode"): "xy_nocross",
            },
        ),
        (
            "D_NONORM",
            "remove normalized attention and use raw exponentiated kernel",
            {
                ("loss", "drift_norm_mode"): "none",
            },
        ),
    ]

    for label, purpose, updates in destructive_specs:
        name = f"ext_{label}_qed_s42"
        cfg = clone_config(
            base_g4,
            name=name,
            output_dir=f"outputs/publication_ext/destructive/{name}",
            seed=42,
            updates=updates,
        )
        add_entry(
            entries,
            "destructive_drift",
            cfg,
            DESTRUCTIVE_DIR / f"{name}.yaml",
            purpose,
        )

    vae_dir = OUT_DIR / "vae_sensitivity"
    vae_dir.mkdir(parents=True, exist_ok=True)
    vae_specs = [
        (
            "V_BETA_LOW",
            "lower beta tests whether a more information-rich latent changes prior quality",
            {
                ("training", "beta"): 0.0025,
            },
        ),
        (
            "V_BETA_HIGH",
            "higher beta tests whether stronger regularization weakens conditional control",
            {
                ("training", "beta"): 0.01,
            },
        ),
        (
            "V_LATENT128",
            "latent dimension sensitivity with a narrower latent bottleneck",
            {
                ("model", "latent_dim"): 128,
            },
        ),
        (
            "V_DEC6",
            "decoder-capacity sensitivity with a deeper decoder",
            {
                ("model", "dec_num_layers"): 6,
            },
        ),
    ]
    for label, purpose, updates in vae_specs:
        name = f"ext_{label}_vae_s42"
        cfg = clone_config(
            base_vae,
            name=name,
            output_dir=f"outputs/publication_ext/vae_sensitivity/{name}",
            seed=42,
            updates=updates,
        )
        add_entry(
            entries,
            "vae_sensitivity",
            cfg,
            vae_dir / f"{name}.yaml",
            purpose,
            module="src.train.train_selfies_vae",
        )

    manifest = {
        "description": "Extension-stage DriftingMol experiment matrix.",
        "notes": [
            "These runs are not part of the audited 8-page AAAI draft.",
            "The destructive_drift group tests whether the V+ - V- construction and normalization are necessary.",
            "The vae_sensitivity group tests beta, latent bottleneck, and decoder-capacity dependence before downstream drift reruns.",
            "Run with: python scripts/run_publication_experiments.py --manifest configs/publication_ext/manifest.json --group destructive_drift",
        ],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(entries)} extension configs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
