#!/usr/bin/env python3
"""Generate reviewer-facing generalization experiment configs.

These runs are intentionally queued behind the active VAE-drift robustness
jobs. They target the two highest-value gaps left after the main QED package:
multi-property seed stability and non-QED single-property control.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "publication_ext" / "generalization"
MANIFEST = ROOT / "configs" / "publication_ext" / "generalization_manifest.json"


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
    purpose: str,
) -> None:
    path = OUT_DIR / f"{cfg['experiment']['name']}.yaml"
    dump_yaml(cfg, path)
    entries.append(
        {
            "group": group,
            "name": cfg["experiment"]["name"],
            "target_property": target_property,
            "config": str(path.relative_to(ROOT)),
            "output_dir": cfg["experiment"]["output_dir"],
            "command": command_for(path),
            "purpose": purpose,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    base_multi4 = load_yaml("configs/publication/pub_G4_multi4_v2_s42.yaml")
    base_qed = load_yaml("configs/final/exp_G4_qed.yaml")

    for seed in (43, 44):
        name = f"ext_G4_multi4_v2_s{seed}"
        cfg = clone_config(
            base_multi4,
            name=name,
            output_dir=f"outputs/publication_ext/generalization/{name}",
            seed=seed,
        )
        add_entry(
            entries,
            "multi4_seed_stability",
            "multi4",
            cfg,
            "additional seed for the four-property no-binning DriftingMol protocol",
        )

    single_property_specs = [
        ("logp", [2], "LogP single-property control with the balanced multi-layer protocol"),
        ("sa_score", [1], "SA-score single-property control with the balanced multi-layer protocol"),
    ]
    for target, prop_indices, purpose in single_property_specs:
        label = target.replace("_score", "")
        name = f"ext_G4_{label}_qed_s42"
        cfg = clone_config(
            base_qed,
            name=name,
            output_dir=f"outputs/publication_ext/generalization/{name}",
            seed=42,
            updates={
                ("data", "prop_indices"): prop_indices,
                ("generator", "cond_dim"): 1,
                ("cond_binning", "enabled"): True,
                ("cond_binning", "n_bins"): 20,
                ("cond_binning", "method"): "quantile",
            },
        )
        add_entry(entries, "single_property_generalization", target, cfg, purpose)

    manifest = {
        "description": "Reviewer-facing generalization experiments queued after VAE-drift robustness runs.",
        "notes": [
            "multi4_seed_stability turns the existing G4 multi4 v2 result into a three-seed stability check.",
            "single_property_generalization tests whether the QED protocol transfers to LogP and SA-score targets.",
            "These jobs should be launched with deferred prerequisites so they do not compete with active VAE-drift runs.",
        ],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(entries)} generalization configs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
