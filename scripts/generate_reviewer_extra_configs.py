#!/usr/bin/env python3
"""Generate high-value reviewer-extra experiment configs.

These jobs are intentionally queued behind the active VAE-drift and
generalization blockers. They fill reviewer gaps that are not covered by the
current four-job generalization queue:

* continuous QED conditioning, to test whether performance depends on QED bins;
* second LogP / SA seeds, to make property-transfer evidence less anecdotal;
* a second downstream drift seed for the strongest live alternative VAE line.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "publication_ext" / "reviewer_extra"
MANIFEST = ROOT / "configs" / "publication_ext" / "reviewer_extra_manifest.json"
GRAPH_CACHE = "data/cache/zinc250k_smiles_cache.pt"


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


def vae_drift_command(config_path: Path, latent_cache: str, vae_ckpt: str, vae_final: str) -> str:
    return (
        f"test -f {vae_final} && "
        f"if [ ! -f {latent_cache} ]; then "
        f"python scripts/build_selfies_latent_cache.py "
        f"--vae_ckpt {vae_ckpt} --graph_cache {GRAPH_CACHE} --output {latent_cache} --batch_size 2048; "
        f"fi && "
        f"python -m src.train.train_selfies_cfg --config {config_path.relative_to(ROOT)}"
    )


def add_entry(
    entries: list[dict[str, Any]],
    group: str,
    target_property: str,
    cfg: dict[str, Any],
    purpose: str,
    display: str,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = OUT_DIR / f"{cfg['experiment']['name']}.yaml"
    dump_yaml(cfg, path)
    entry = {
        "group": group,
        "name": cfg["experiment"]["name"],
        "display": display,
        "target_property": target_property,
        "config": str(path.relative_to(ROOT)),
        "output_dir": cfg["experiment"]["output_dir"],
        "command": command or command_for(path),
        "purpose": purpose,
    }
    if extra:
        entry.update(extra)
    entries.append(entry)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    base_qed = load_yaml("configs/final/exp_G4_qed.yaml")

    qed_cont = clone_config(
        base_qed,
        name="ext_G4_qed_continuous_s42",
        output_dir="outputs/publication_ext/reviewer_extra/ext_G4_qed_continuous_s42",
        seed=42,
        updates={
            ("cond_binning", "enabled"): False,
            ("cfg", "positive_mode"): "prop",
        },
    )
    add_entry(
        entries,
        group="continuous_conditioning",
        target_property="qed",
        cfg=qed_cont,
        display="QED continuous control",
        purpose="tests whether QED control is caused by quantile-bin conditioning rather than continuous target guidance",
    )

    single_property_specs = [
        (
            "ext_G4_logp_qed_s43",
            "logp",
            [2],
            43,
            "LogP transfer, seed 43",
            "second-seed LogP transfer check for single-property generalization",
        ),
        (
            "ext_G4_sa_qed_s43",
            "sa_score",
            [1],
            43,
            "SA transfer, seed 43",
            "second-seed SA-score transfer check for single-property generalization",
        ),
    ]
    for name, target, prop_indices, seed, display, purpose in single_property_specs:
        cfg = clone_config(
            base_qed,
            name=name,
            output_dir=f"outputs/publication_ext/reviewer_extra/{name}",
            seed=seed,
            updates={
                ("data", "prop_indices"): prop_indices,
                ("generator", "cond_dim"): 1,
                ("cond_binning", "enabled"): True,
                ("cond_binning", "n_bins"): 20,
                ("cond_binning", "method"): "quantile",
            },
        )
        add_entry(
            entries,
            group="single_property_seed_extension",
            target_property=target,
            cfg=cfg,
            display=display,
            purpose=purpose,
        )

    lowbeta_ckpt = "outputs/publication_ext/vae_sensitivity/ext_V_BETA_LOW_vae_s42/best.pt"
    lowbeta_final = "outputs/publication_ext/vae_sensitivity/ext_V_BETA_LOW_vae_s42/final_metrics.json"
    lowbeta_cache = "data/cache/zinc_latent_cache_ext_vae_lowbeta_s42.pt"
    lowbeta_seed = clone_config(
        base_qed,
        name="ext_vae_lowbeta_drift_qed_s43",
        output_dir="outputs/publication_ext/reviewer_extra/ext_vae_lowbeta_drift_qed_s43",
        seed=43,
        updates={
            ("vae", "checkpoint"): lowbeta_ckpt,
            ("data", "latent_cache_path"): lowbeta_cache,
            ("feature_space", "micro_batch"): 128,
        },
    )
    lowbeta_path = OUT_DIR / f"{lowbeta_seed['experiment']['name']}.yaml"
    add_entry(
        entries,
        group="vae_drift_seed_extension",
        target_property="qed",
        cfg=lowbeta_seed,
        display="Low-beta VAE drift, seed 43",
        purpose="second generator seed for the strongest live alternative-VAE downstream drift line",
        command=vae_drift_command(lowbeta_path, lowbeta_cache, lowbeta_ckpt, lowbeta_final),
        extra={
            "depends_on": [lowbeta_final],
            "vae_run": "ext_V_BETA_LOW_vae_s42",
            "vae_checkpoint": lowbeta_ckpt,
            "latent_cache": lowbeta_cache,
        },
    )

    payload = {
        "description": "Additional reviewer-facing experiments queued after the current VAE-drift and generalization blockers.",
        "notes": [
            "These entries are intentionally not launched until their predecessor jobs finish.",
            "The set targets three high-value risks: QED bin dependence, single-property transfer seed stability, and alternative-VAE downstream seed stability.",
            "They are extension evidence and should be interpreted separately from the locked 8-page draft until complete.",
        ],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(entries)} reviewer-extra configs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
