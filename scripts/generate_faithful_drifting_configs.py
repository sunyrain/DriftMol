#!/usr/bin/env python3
"""Generate reviewer-facing Drifting-faithfulness experiment configs.

This matrix is intentionally separate from configs/publication_ext because it
answers a narrower reviewer question: did we faithfully reproduce the original
Drifting Models algorithm before adding molecule-specific decoder coupling?
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "reviewer_faithful"
CORE_DIR = OUT_DIR / "core"
ALLOC_DIR = OUT_DIR / "allocation"
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
    updates: dict[tuple[str, ...], Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    deep_set(cfg, ("experiment", "name"), name)
    deep_set(cfg, ("experiment", "output_dir"), output_dir)
    deep_set(cfg, ("experiment", "seed"), 42)
    for key_path, value in (updates or {}).items():
        deep_set(cfg, key_path, value)
    return cfg


def command_for(path: Path) -> str:
    return f"python -m src.train.train_selfies_cfg --config {path.relative_to(ROOT)}"


def add_entry(
    entries: list[dict[str, Any]],
    group: str,
    cfg: dict[str, Any],
    path: Path,
    purpose: str,
) -> None:
    dump_yaml(cfg, path)
    entries.append(
        {
            "group": group,
            "name": cfg["experiment"]["name"],
            "config": str(path.relative_to(ROOT)),
            "output_dir": cfg["experiment"]["output_dir"],
            "command": command_for(path),
            "purpose": purpose,
        }
    )


def paper_faithful_updates() -> dict[tuple[str, ...], Any]:
    """Common settings mirroring the original Drifting ablation protocol.

    Original Drifting Models ablations use:
    - latent generation,
    - pretrained latent-MAE feature extractor,
    - Algorithm-2 V = V+ - V- with bidirectional softmax and cross weights,
    - generated samples as negatives,
    - Appendix-A.6 feature-distance and drift normalization,
    - N_c=64, N_pos=64, N_neg=64, B=4096,
    - temperatures {0.02, 0.05, 0.2},
    - training-time CFG alpha in [1, 4] sampled with alpha^-3,
    - no molecule-specific z-diversity add-on.

    The molecular analogue uses QED quantile bins as class labels.
    """
    return {
        ("training", "epochs"): 100,
        ("training", "warmup_epochs"): 5,
        ("loss", "lambda_drift"): 1.0,
        ("loss", "lambda_decoupled_drift"): 0.0,
        ("loss", "lambda_zdrift"): 0.0,
        ("loss", "lambda_dec_drift"): 0.0,
        ("loss", "lambda_zdiv"): 0.0,
        ("loss", "lambda_phidiv"): 0.0,
        ("loss", "lambda_prop"): 0.0,
        ("loss", "temperatures"): [0.02, 0.05, 0.2],
        ("loss", "drift_normalize"): True,
        ("loss", "drift_normalize_dist"): True,
        ("loss", "drift_normalize_mode"): "batch",
        ("loss", "drift_norm_mode"): "xy",
        ("loss", "drift_attraction_scale"): 1.0,
        ("loss", "drift_repulsion_scale"): 1.0,
        ("loss", "stop_grad_drift"): False,
        ("cfg", "n_groups"): 64,
        ("cfg", "n_gen"): 64,
        ("cfg", "n_pos"): 64,
        ("cfg", "n_unc"): 16,
        ("cfg", "positive_mode"): "prop",
        ("cfg", "alpha_power"): 3,
        ("cfg", "alpha_min"): 1.0,
        ("cfg", "alpha_max"): 4.0,
        ("cond_binning", "enabled"): True,
        ("cond_binning", "n_bins"): 20,
        ("cond_binning", "method"): "quantile",
        ("selection", "eval_every_epochs"): 5,
        ("selection", "num_generated_samples"): 2000,
        ("selection", "sample_batch_size"): 512,
        ("eval", "num_generated_samples"): 10000,
        ("eval", "alpha_sweep"): [1.0, 1.5, 2.0, 3.0, 5.0],
    }


def merge_updates(*parts: dict[tuple[str, ...], Any]) -> dict[tuple[str, ...], Any]:
    out: dict[tuple[str, ...], Any] = {}
    for part in parts:
        out.update(part)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    ALLOC_DIR.mkdir(parents=True, exist_ok=True)

    base_phi = load_yaml("configs/final_phi/exp_C1_qed.yaml")
    base_z = load_yaml("configs/final/exp_A2_qed.yaml")
    strict = paper_faithful_updates()

    entries: list[dict[str, Any]] = []

    core_specs = [
        (
            "rf_FD_STRICT_PLAIN_PHI_QED_s42",
            base_phi,
            merge_updates(
                strict,
                {
                    ("feature_space", "mode"): "phi",
                    ("phi", "checkpoint"): "outputs/foundation/zinc_phi_plain/best_latent_mae.pt",
                },
            ),
            "Strict molecular analogue of original Drifting: latent-MAE phi, Algorithm 2, generated negatives, QED bins as classes.",
        ),
        (
            "rf_FD_STRICT_PROP_PHI_QED_s42",
            base_phi,
            merge_updates(
                strict,
                {
                    ("feature_space", "mode"): "phi",
                    ("phi", "checkpoint"): "outputs/foundation/zinc_phi_prop/best_latent_mae.pt",
                },
            ),
            "Feature-quality check mirroring latent-MAE + classifier fine-tuning: property-aware phi, same Drifting protocol.",
        ),
        (
            "rf_FD_STRICT_RANDOM_PHI_QED_s42",
            base_phi,
            merge_updates(
                strict,
                {
                    ("feature_space", "mode"): "random",
                    ("phi", "checkpoint"): "",
                },
            ),
            "Random frozen phi control: tests whether Algorithm 2 needs a meaningful feature metric.",
        ),
        (
            "rf_FD_STRICT_ZSPACE_QED_s42",
            base_z,
            merge_updates(
                strict,
                {
                    ("loss", "lambda_drift"): 0.0,
                    ("loss", "lambda_zdrift"): 1.0,
                    ("loss", "zdrift_temperatures"): [0.02, 0.05, 0.2],
                    ("feature_space", "mode"): "phi",
                    ("phi", "checkpoint"): "",
                },
            ),
            "No-feature z-space control: molecular analogue of the original paper's feature-extractor necessity claim.",
        ),
    ]

    for name, base, updates, purpose in core_specs:
        cfg = clone_config(
            base,
            name=name,
            output_dir=f"outputs/reviewer_faithful/core/{name}",
            updates=updates,
        )
        add_entry(entries, "faithful_core", cfg, CORE_DIR / f"{name}.yaml", purpose)

    allocation_specs = [
        ("rf_FD_ALLOC_POS01_QED_s42", {"n_groups": 64, "n_gen": 64, "n_pos": 1, "n_unc": 16},
         "Positive-sample allocation ablation: Npos=1 with fixed N_c=64, Nneg=64."),
        ("rf_FD_ALLOC_POS16_QED_s42", {"n_groups": 64, "n_gen": 64, "n_pos": 16, "n_unc": 16},
         "Positive-sample allocation ablation: Npos=16 with fixed N_c=64, Nneg=64."),
        ("rf_FD_ALLOC_POS32_QED_s42", {"n_groups": 64, "n_gen": 64, "n_pos": 32, "n_unc": 16},
         "Positive-sample allocation ablation: Npos=32 with fixed N_c=64, Nneg=64."),
        ("rf_FD_ALLOC_POS64_QED_s42", {"n_groups": 64, "n_gen": 64, "n_pos": 64, "n_unc": 16},
         "Positive-sample allocation ablation baseline: Npos=64 with fixed N_c=64, Nneg=64."),
        ("rf_FD_ALLOC_NEG16_QED_s42", {"n_groups": 256, "n_gen": 16, "n_pos": 16, "n_unc": 16},
         "Negative-sample allocation ablation: B=4096 with Nneg=16."),
        ("rf_FD_ALLOC_NEG32_QED_s42", {"n_groups": 128, "n_gen": 32, "n_pos": 32, "n_unc": 16},
         "Negative-sample allocation ablation: B=4096 with Nneg=32."),
    ]

    for name, batch_cfg, purpose in allocation_specs:
        updates = merge_updates(
            strict,
            {
                ("feature_space", "mode"): "phi",
                ("phi", "checkpoint"): "outputs/foundation/zinc_phi_plain/best_latent_mae.pt",
                ("cfg", "n_groups"): batch_cfg["n_groups"],
                ("cfg", "n_gen"): batch_cfg["n_gen"],
                ("cfg", "n_pos"): batch_cfg["n_pos"],
                ("cfg", "n_unc"): batch_cfg["n_unc"],
            },
        )
        cfg = clone_config(
            base_phi,
            name=name,
            output_dir=f"outputs/reviewer_faithful/allocation/{name}",
            updates=updates,
        )
        add_entry(entries, "faithful_allocation", cfg, ALLOC_DIR / f"{name}.yaml", purpose)

    manifest = {
        "description": "Reviewer-facing faithful Drifting reproduction matrix for DriftingMol.",
        "notes": [
            "These runs are supplemental. The strict core queue is deferred until destructive ablations finish, while VAE sensitivity can continue on GPU1.",
            "They isolate whether the original Drifting Models recipe works in the SELFIES latent setting before decoder-coupled modifications.",
            "The strict core setting mirrors Algorithm 2 plus Appendix-A.6 normalization, latent-MAE phi, generated negatives, N_c=64, N_pos=64, N_neg=64, and tau={0.02,0.05,0.2}.",
            "QED quantile bins are used as class labels, which is the molecular analogue of ImageNet class conditioning.",
        ],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(entries)} faithful Drifting configs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
