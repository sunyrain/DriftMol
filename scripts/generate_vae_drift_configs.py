#!/usr/bin/env python3
"""Generate downstream drifting configs for alternative SELFIES VAEs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "publication_ext" / "vae_drift"
MANIFEST = ROOT / "configs" / "publication_ext" / "vae_drift_manifest.json"
BASE_CONFIG = ROOT / "configs" / "publication" / "pub_G4_qed_s42.yaml"
GRAPH_CACHE = "data/cache/zinc250k_smiles_cache.pt"


SPECS = [
    {
        "name": "ext_vae_lowbeta_drift_qed_s42",
        "display": "Low-beta VAE",
        "vae_run": "ext_V_BETA_LOW_vae_s42",
        "vae_checkpoint": "outputs/publication_ext/vae_sensitivity/ext_V_BETA_LOW_vae_s42/best.pt",
        "vae_final_metrics": "outputs/publication_ext/vae_sensitivity/ext_V_BETA_LOW_vae_s42/final_metrics.json",
        "latent_cache": "data/cache/zinc_latent_cache_ext_vae_lowbeta_s42.pt",
        "purpose": "downstream DriftingMol check using the completed low-beta SELFIES VAE",
    },
    {
        "name": "ext_vae_highbeta_drift_qed_s42",
        "display": "High-beta VAE",
        "vae_run": "ext_V_BETA_HIGH_vae_s42",
        "vae_checkpoint": "outputs/publication_ext/vae_sensitivity/ext_V_BETA_HIGH_vae_s42/best.pt",
        "vae_final_metrics": "outputs/publication_ext/vae_sensitivity/ext_V_BETA_HIGH_vae_s42/final_metrics.json",
        "latent_cache": "data/cache/zinc_latent_cache_ext_vae_highbeta_s42.pt",
        "purpose": "downstream DriftingMol check using the completed high-beta SELFIES VAE",
    },
    {
        "name": "ext_vae_latent128_drift_qed_s42",
        "display": "Latent-128 VAE",
        "vae_run": "ext_V_LATENT128_vae_s42",
        "vae_checkpoint": "outputs/publication_ext/vae_sensitivity/ext_V_LATENT128_vae_s42/best.pt",
        "vae_final_metrics": "outputs/publication_ext/vae_sensitivity/ext_V_LATENT128_vae_s42/final_metrics.json",
        "latent_cache": "data/cache/zinc_latent_cache_ext_vae_latent128_s42.pt",
        "purpose": "downstream DriftingMol architecture check using a narrower latent bottleneck",
    },
    {
        "name": "ext_vae_dec6_drift_qed_s42",
        "display": "Decoder-6 VAE",
        "vae_run": "ext_V_DEC6_vae_s42",
        "vae_checkpoint": "outputs/publication_ext/vae_sensitivity/ext_V_DEC6_vae_s42/best.pt",
        "vae_final_metrics": "outputs/publication_ext/vae_sensitivity/ext_V_DEC6_vae_s42/final_metrics.json",
        "latent_cache": "data/cache/zinc_latent_cache_ext_vae_dec6_s42.pt",
        "purpose": "downstream DriftingMol architecture check using a deeper decoder",
    },
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def deep_set(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cur = data
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def command_for(config_path: Path, spec: dict[str, str]) -> str:
    cfg_rel = config_path.relative_to(ROOT)
    cache = spec["latent_cache"]
    ckpt = spec["vae_checkpoint"]
    final_metrics = spec["vae_final_metrics"]
    return (
        f"test -f {final_metrics} && "
        f"if [ ! -f {cache} ]; then "
        f"python scripts/build_selfies_latent_cache.py "
        f"--vae_ckpt {ckpt} --graph_cache {GRAPH_CACHE} --output {cache} --batch_size 2048; "
        f"fi && "
        f"python -m src.train.train_selfies_cfg --config {cfg_rel}"
    )


def main() -> None:
    base = read_yaml(BASE_CONFIG)
    entries: list[dict[str, Any]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for spec in SPECS:
        cfg = copy.deepcopy(base)
        deep_set(cfg, ("vae", "checkpoint"), spec["vae_checkpoint"])
        deep_set(cfg, ("data", "latent_cache_path"), spec["latent_cache"])
        deep_set(cfg, ("feature_space", "micro_batch"), 128)
        deep_set(cfg, ("experiment", "name"), spec["name"])
        deep_set(cfg, ("experiment", "output_dir"), f"outputs/publication_ext/vae_drift/{spec['name']}")
        deep_set(cfg, ("experiment", "seed"), 42)

        config_path = OUT_DIR / f"{spec['name']}.yaml"
        dump_yaml(cfg, config_path)
        entries.append(
            {
                "group": "vae_drift_downstream",
                "name": spec["name"],
                "display": spec["display"],
                "config": str(config_path.relative_to(ROOT)),
                "output_dir": cfg["experiment"]["output_dir"],
                "command": command_for(config_path, spec),
                "purpose": spec["purpose"],
                "depends_on": [spec["vae_final_metrics"]],
                "vae_run": spec["vae_run"],
                "vae_checkpoint": spec["vae_checkpoint"],
                "latent_cache": spec["latent_cache"],
            }
        )

    payload = {
        "description": "Downstream DriftingMol checks under alternative SELFIES VAE checkpoints.",
        "notes": [
            "Each entry first builds a latent cache for the selected VAE if it is missing.",
            "These runs test whether QED drifting depends on one SELFIES VAE architecture or beta setting.",
            "They are extension-stage reviewer experiments and should be interpreted separately from the locked 8-page draft until complete.",
        ],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(entries)} downstream VAE drift configs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
