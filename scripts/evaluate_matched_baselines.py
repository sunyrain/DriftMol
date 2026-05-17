#!/usr/bin/env python3
"""Evaluate lightweight protocol-matched QED baselines.

These baselines are deliberately simple and reproducible:

* Retrieval: sample training molecules from the target QED quantile bin.
* VAE-jitter: sample a training latent from the target bin, add Gaussian noise,
  and decode with the frozen SELFIES VAE.
* Bin-Gaussian: fit a diagonal latent Gaussian inside each target QED bin and
  decode samples from that conditional latent prior.

They use the same 20 target bins and 10k total conditional samples as the main
QED evaluation, but they are not trained generators.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_JSON = ROOT / "results" / "matched_baselines_qed.json"
OUT_TEX = ROOT / "results" / "tables" / "tab_qed_matched_baselines.tex"

from src.models.selfies_vae import SelfiesVAE, SelfiesVAEConfig, set_vocab

RDLogger.DisableLog("rdApp.warning")


def canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles((smi or "").strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_vae(device: torch.device) -> SelfiesVAE:
    ckpt = torch.load(ROOT / "outputs/foundation/zinc_selfies_vae_v2/best.pt", map_location=device)
    set_vocab(ckpt["vocab"])
    cfg = SelfiesVAEConfig(**ckpt["vae_cfg"])
    model = SelfiesVAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_qed_bins(train_qed: torch.Tensor, n_bins: int) -> tuple[list[torch.Tensor], list[float]]:
    mean = train_qed.mean()
    std = train_qed.std().clamp(min=1e-6)
    norm = (train_qed - mean) / std
    sorted_vals, _ = norm.sort()
    n_total = norm.numel()
    edges = [sorted_vals[0].item()]
    for b in range(1, n_bins):
        edges.append(sorted_vals[b * n_total // n_bins].item())
    edges.append(sorted_vals[-1].item() + 1e-6)

    bins = []
    centers = []
    for b in range(n_bins):
        mask = (norm >= edges[b]) & (norm < edges[b + 1])
        idx = mask.nonzero(as_tuple=True)[0]
        bins.append(idx)
        centers.append(float(train_qed[idx].mean().item()))
    return bins, centers


def molecular_metrics(smiles: list[str], targets: list[float], train_canon: set[str]) -> dict:
    valid = []
    actual = []
    for smi, target in zip(smiles, targets):
        can = canonicalize(smi)
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue
        valid.append((can, mol, target))
        actual.append(float(QED.qed(mol)))

    canonical = [item[0] for item in valid]
    unique = sorted(set(canonical))
    targets_valid = np.array([item[2] for item in valid], dtype=float)
    actuals = np.array(actual, dtype=float)
    validity = len(valid) / max(len(smiles), 1)
    uniqueness = len(unique) / max(len(valid), 1)
    novelty = sum(1 for smi in unique if smi not in train_canon) / max(len(unique), 1)

    result = {
        "num_samples": len(smiles),
        "num_valid": len(valid),
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
    }
    if len(actuals) >= 10:
        rho, pval = spearmanr(targets_valid, actuals)
        pearson, _ = pearsonr(targets_valid, actuals)
        slope, intercept = np.polyfit(targets_valid, actuals, 1)
        abs_err = np.abs(targets_valid - actuals)
        result.update(
            {
                "spearman_rho": float(rho),
                "spearman_pval": float(pval),
                "pearson_r": float(pearson),
                "mae": float(abs_err.mean()),
                "slope": float(slope),
                "intercept": float(intercept),
                "success_0p05": float((abs_err <= 0.05).mean()),
                "success_0p10": float((abs_err <= 0.10).mean()),
                "target_mean": float(targets_valid.mean()),
                "actual_mean": float(actuals.mean()),
            }
        )

    try:
        mols_for_div = [item[1] for item in valid[:500]]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols_for_div]
        if len(fps) >= 2:
            sim_sum = 0.0
            sim_count = 0
            rng = random.Random(123)
            n_pairs = min(5000, len(fps) * (len(fps) - 1) // 2)
            for _ in range(n_pairs):
                i, j = rng.sample(range(len(fps)), 2)
                sim_sum += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sim_count += 1
            result["int_div"] = float(1.0 - sim_sum / sim_count)
    except Exception:
        pass

    try:
        scaffolds = set()
        for _, mol, _ in valid:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.add(Chem.MolToSmiles(core))
        result["scaffold_diversity"] = len(scaffolds) / max(len(valid), 1)
        result["n_unique_scaffolds"] = len(scaffolds)
    except Exception:
        pass

    per_target_u = []
    valid_targets = [item[2] for item in valid]
    for target in sorted(set(round(t, 6) for t in valid_targets)):
        smis = [smi for smi, t in zip(canonical, valid_targets) if round(t, 6) == target]
        if smis:
            per_target_u.append(len(set(smis)) / len(smis))
    if per_target_u:
        result["min_bin_uniqueness"] = float(min(per_target_u))

    return result


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "---"
    try:
        if math.isnan(float(value)):
            return "---"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "---" if value is None else f"{100.0 * float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--jitter-std", type=float, default=0.10)
    parser.add_argument("--gaussian-std-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = torch.load(ROOT / "data/cache/zinc_latent_cache_v2.pt", map_location="cpu")
    train_z = cache["train"]
    train_qed = cache["train_props"][:, 0]
    train_smiles = [str(s).strip() for s in cache["train_smiles"]]
    train_canon = {can for smi in train_smiles if (can := canonicalize(smi)) is not None}
    bins, centers = build_qed_bins(train_qed, args.n_bins)
    per_bin = args.num_samples // args.n_bins

    retrieval_smiles: list[str] = []
    retrieval_targets: list[float] = []
    chosen_indices: list[int] = []
    for bin_idx, target in enumerate(centers):
        candidates = bins[bin_idx].numpy()
        replace = len(candidates) < per_bin
        sampled = rng.choice(candidates, size=per_bin, replace=replace)
        chosen_indices.extend(int(i) for i in sampled)
        retrieval_smiles.extend(train_smiles[int(i)] for i in sampled)
        retrieval_targets.extend([target] * per_bin)

    retrieval = molecular_metrics(retrieval_smiles, retrieval_targets, train_canon)

    device = torch.device(args.device)
    vae = load_vae(device)
    jitter_smiles: list[str] = []
    jitter_targets: list[float] = []
    with torch.no_grad():
        for start in range(0, len(chosen_indices), 512):
            idx = torch.tensor(chosen_indices[start : start + 512], dtype=torch.long)
            z = train_z[idx].to(device)
            if args.jitter_std > 0:
                z = z + args.jitter_std * torch.randn_like(z)
            jitter_smiles.extend(vae.sample_smiles(z, temperature=0.0))
            jitter_targets.extend(retrieval_targets[start : start + 512])

    jitter = molecular_metrics(jitter_smiles, jitter_targets, train_canon)

    gaussian_smiles: list[str] = []
    gaussian_targets: list[float] = []
    with torch.no_grad():
        for bin_idx, target in enumerate(centers):
            idx = bins[bin_idx]
            z_bin = train_z[idx]
            mean = z_bin.mean(dim=0)
            std = z_bin.std(dim=0).clamp(min=1e-3) * args.gaussian_std_scale
            remaining = per_bin
            while remaining > 0:
                batch = min(512, remaining)
                z = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(batch, z_bin.shape[1])
                gaussian_smiles.extend(vae.sample_smiles(z.to(device), temperature=0.0))
                gaussian_targets.extend([target] * batch)
                remaining -= batch

    gaussian = molecular_metrics(gaussian_smiles, gaussian_targets, train_canon)

    output = {
        "protocol": {
            "num_samples": args.num_samples,
            "n_bins": args.n_bins,
            "samples_per_bin": per_bin,
            "jitter_std": args.jitter_std,
            "gaussian_std_scale": args.gaussian_std_scale,
            "seed": args.seed,
            "target_property": "qed",
        },
        "target_centers": centers,
        "baselines": {
            "retrieval": retrieval,
            "vae_jitter": jitter,
            "bin_gaussian": gaussian,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2))

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Retrieval", retrieval),
        (f"VAE-jitter ($\\sigma={args.jitter_std}$)", jitter),
        (f"Bin-Gaussian ($s={args.gaussian_std_scale}$)", gaussian),
    ]
    lines = [
        "% Generated by scripts/evaluate_matched_baselines.py",
        "\\begin{tabular}{l c c c c c c c}",
        "\\toprule",
        "Baseline & $\\rho$ & MAE & Slope & U (\\%) & N (\\%) & IntDiv & Succ@0.10 \\\\",
        "\\midrule",
    ]
    for label, metrics in rows:
        lines.append(
            f"{label} & {fmt(metrics.get('spearman_rho'))} & {fmt(metrics.get('mae'))} & "
            f"{fmt(metrics.get('slope'))} & {pct(metrics.get('uniqueness'))} & "
            f"{pct(metrics.get('novelty'))} & {fmt(metrics.get('int_div'))} & "
            f"{pct(metrics.get('success_0p10'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    OUT_TEX.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")
    for label, metrics in rows:
        print(label, {k: metrics.get(k) for k in ["spearman_rho", "mae", "slope", "uniqueness", "novelty", "int_div", "success_0p10"]})


if __name__ == "__main__":
    main()
