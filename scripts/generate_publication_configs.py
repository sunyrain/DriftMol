#!/usr/bin/env python3
"""Generate publication-stage experiment configs and a runnable manifest."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "publication"
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
    seed: int | None = None,
    updates: dict[tuple[str, ...], Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    deep_set(cfg, ("experiment", "name"), name)
    deep_set(cfg, ("experiment", "output_dir"), output_dir)
    if seed is not None:
        deep_set(cfg, ("experiment", "seed"), seed)
    for key_path, value in (updates or {}).items():
        deep_set(cfg, key_path, value)
    return cfg


def command_for(config_path: Path) -> str:
    rel = config_path.relative_to(ROOT)
    return f"python -m src.train.train_selfies_cfg --config {rel}"


def add_entry(entries: list[dict[str, Any]], group: str, cfg: dict[str, Any], filename: str) -> None:
    path = OUT_DIR / filename
    dump_yaml(cfg, path)
    entries.append({
        "group": group,
        "name": cfg["experiment"]["name"],
        "config": str(path.relative_to(ROOT)),
        "output_dir": cfg["experiment"]["output_dir"],
        "command": command_for(path),
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    base_f = load_yaml("configs/final/exp_F_qed.yaml")
    base_a6 = load_yaml("configs/final/exp_A6_qed.yaml")
    base_a8 = load_yaml("configs/final/exp_A8_qed.yaml")
    base_g4 = load_yaml("configs/final/exp_G4_qed.yaml")

    # 1. Minimal multi-seed set for confidence intervals.
    seed_bases = {
        "F": base_f,
        "A6": base_a6,
        "A8": base_a8,
        "G4": base_g4,
    }
    for variant, base in seed_bases.items():
        for seed in (42, 43, 44):
            name = f"pub_{variant}_qed_s{seed}"
            cfg = clone_config(
                base,
                name=name,
                output_dir=f"outputs/publication/seeds/{name}",
                seed=seed,
            )
            add_entry(entries, "qed_3seed", cfg, f"{name}.yaml")

    # 2. Paper-faithful audit: isolate deviations from DriftingModels.pdf.
    paperish_common = {
        ("loss", "temperatures"): [0.02, 0.05, 0.2],
        ("cfg", "n_groups"): 64,
        ("cfg", "n_gen"): 64,
        ("cfg", "n_pos"): 64,
        ("cfg", "n_unc"): 16,
        ("cfg", "positive_mode"): "prop",
        ("cfg", "alpha_power"): 3,
        ("cond_binning", "enabled"): True,
        ("cond_binning", "n_bins"): 20,
    }
    audit_specs = {
        "P1_paper_tau_batch_lambda": {
            **paperish_common,
            ("loss", "drift_normalize_mode"): "batch",
        },
        "P2_paper_tau_fixed_lambda": {
            **paperish_common,
            ("loss", "drift_normalize_mode"): "fixed",
        },
        "P3_no_cfg": {
            **paperish_common,
            ("loss", "drift_normalize_mode"): "fixed",
            ("cfg", "alpha_min"): 1.0,
            ("cfg", "alpha_max"): 1.0,
            ("cfg", "n_unc"): 0,
            ("generator", "p_uncond"): 0.0,
        },
        "P4_no_zdiv": {
            **paperish_common,
            ("loss", "drift_normalize_mode"): "fixed",
            ("loss", "lambda_zdiv"): 0.0,
        },
        "P5_y_only_norm": {
            **paperish_common,
            ("loss", "drift_normalize_mode"): "fixed",
            ("loss", "drift_norm_mode"): "y",
        },
        "P6_no_cross_norm": {
            **paperish_common,
            ("loss", "drift_normalize_mode"): "fixed",
            ("loss", "drift_norm_mode"): "y_nocross",
        },
    }
    for label, updates in audit_specs.items():
        name = f"pub_{label}_qed_s42"
        cfg = clone_config(
            base_f,
            name=name,
            output_dir=f"outputs/publication/audit/{name}",
            seed=42,
            updates=updates,
        )
        add_entry(entries, "paper_faithfulness_audit", cfg, f"{name}.yaml")

    # 3. Diversity-control Pareto front around the current strongest recipe.
    for zdiv in (0.0, 0.5, 1.0, 2.0, 4.0):
        zlabel = str(zdiv).replace(".", "p")
        name = f"pub_G4_qed_zdiv{zlabel}_s42"
        cfg = clone_config(
            base_g4,
            name=name,
            output_dir=f"outputs/publication/zdiv/{name}",
            seed=42,
            updates={("loss", "lambda_zdiv"): zdiv},
        )
        add_entry(entries, "zdiv_pareto", cfg, f"{name}.yaml")

    # 4. Fair multi-property rerun using the v2 protocol.
    name = "pub_G4_multi4_v2_s42"
    cfg = clone_config(
        base_g4,
        name=name,
        output_dir=f"outputs/publication/multi4/{name}",
        seed=42,
        updates={
            ("data", "prop_indices"): [0, 1, 2, 3],
            ("generator", "cond_dim"): 4,
            ("cond_binning", "enabled"): False,
            ("cfg", "positive_mode"): "prop",
        },
    )
    add_entry(entries, "multi4_v2", cfg, f"{name}.yaml")

    manifest = {
        "description": "Publication-stage DriftingMol experiment matrix.",
        "notes": [
            "qed_3seed gives confidence intervals for the main QED table.",
            "paper_faithfulness_audit isolates deviations from DriftingModels.pdf.",
            "zdiv_pareto quantifies the rho/uniqueness tradeoff.",
            "multi4_v2 avoids the legacy QED-only binning protocol.",
        ],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(entries)} configs to {OUT_DIR}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
