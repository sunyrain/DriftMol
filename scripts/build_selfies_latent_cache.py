"""
Build latent cache from a trained SELFIES VAE.

Encodes all QM9 molecules through the frozen SELFIES VAE encoder,
saves z-vectors + molecular properties for downstream Latent-MAE
and Drifting Generator training.

Usage:
    python scripts/build_selfies_latent_cache.py \
        --vae_ckpt outputs/selfies_vae/best.pt \
        --output data/cache/selfies_latent_cache.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.selfies_vae import (
    SelfiesVAE, SelfiesVAEConfig,
    set_vocab, smiles_to_token_ids, PAD_IDX,
)
from src.models.selfies_vae_spatial import (
    SelfiesSpatialVAE, SelfiesSpatialVAEConfig,
)


def compute_mol_properties(smiles_list: list[str]) -> torch.Tensor:
    """Compute [QED, SA, LogP, MolWt] for a list of SMILES. Invalid → NaN."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as QED_module
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        import importlib, os, glob
        sascorer = None
        for p in glob.glob('/root/miniconda3/share/RDKit/Contrib/SA_Score/sascorer.py'):
            spec = importlib.util.spec_from_file_location("sascorer", p)
            sascorer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sascorer)
            break

    props = np.full((len(smiles_list), 4), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            props[i, 0] = QED_module.qed(mol)
        except Exception:
            pass
        try:
            props[i, 1] = sascorer.calculateScore(mol) if sascorer else float("nan")
        except Exception:
            pass
        try:
            props[i, 2] = Descriptors.MolLogP(mol)
        except Exception:
            pass
        try:
            props[i, 3] = Descriptors.MolWt(mol)
        except Exception:
            pass
    return torch.from_numpy(props)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--graph_cache", default="data/cache/qm9_graph_cache.pt",
                        help="QM9 graph cache for SMILES list")
    parser.add_argument("--output", default="data/cache/selfies_latent_cache.pt")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--reparam", action="store_true",
                        help="Store reparameterized z=mu+sigma*eps instead of mu only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load VAE checkpoint ──
    print("[vae] Loading checkpoint...")
    ckpt = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    vae_cfg_dict = ckpt["vae_cfg"]
    vocab = ckpt["vocab"]

    # Restore vocabulary
    set_vocab(vocab)

    # Build model (support both flat and spatial)
    model_type = ckpt.get("model_type", "flat")
    if model_type == "spatial":
        vae_cfg = SelfiesSpatialVAEConfig()
        for k, v in vae_cfg_dict.items():
            if hasattr(vae_cfg, k):
                setattr(vae_cfg, k, v)
        vae_cfg.vocab_size = len(vocab)
        vae = SelfiesSpatialVAE(vae_cfg).to(device)
        print(f"  [spatial] latent={vae_cfg.latent_channels}×{vae_cfg.latent_height}×"
              f"{vae_cfg.latent_width}={vae_cfg.latent_dim}d, vocab={len(vocab)}")
    else:
        vae_cfg = SelfiesVAEConfig(**vae_cfg_dict)
        vae = SelfiesVAE(vae_cfg).to(device)
        print(f"  [flat] latent_dim={vae_cfg.latent_dim}, vocab={len(vocab)}")

    vae.load_state_dict(ckpt["model_state"])
    vae.eval()

    # ── Load SMILES ──
    print("[data] Loading QM9 SMILES...")
    graph_cache = torch.load(args.graph_cache, map_location="cpu")
    all_smiles: list[str] = graph_cache["smiles"]
    print(f"  Total: {len(all_smiles)}")

    # ── Tokenize ──
    max_len = vae_cfg.max_len
    token_data = []
    valid_indices = []
    for i, smi in enumerate(all_smiles):
        ids = smiles_to_token_ids(smi, max_len)
        if ids is not None:
            token_data.append(ids)
            valid_indices.append(i)
    data_tensor = torch.stack(token_data)
    valid_smiles = [all_smiles[i] for i in valid_indices]
    print(f"  Tokenized: {len(token_data)} / {len(all_smiles)}")

    # ── Train/val/test split (same seed as training) ──
    N = len(data_tensor)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))

    # Use 80/10/10 split to match graph VAE convention
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    splits = {
        "train": data_tensor[train_idx],
        "val": data_tensor[val_idx],
        "test": data_tensor[test_idx],
    }
    splits_smiles = {
        "train": [valid_smiles[i] for i in train_idx.tolist()],
        "val": [valid_smiles[i] for i in val_idx.tolist()],
        "test": [valid_smiles[i] for i in test_idx.tolist()],
    }

    # ── Encode all splits ──
    use_reparam = getattr(args, 'reparam', False)
    print(f"[encode] reparameterize={use_reparam}")
    z_splits = {}
    for split_name, split_data in splits.items():
        loader = DataLoader(TensorDataset(split_data), batch_size=args.batch_size, shuffle=False)
        z_list = []
        for (batch,) in loader:
            batch = batch.to(device)
            mu, logvar = vae.encode(batch)
            if use_reparam:
                std = (logvar * 0.5).exp()
                z = mu + std * torch.randn_like(std)
                z_list.append(z.cpu())
            else:
                z_list.append(mu.cpu())
        z_all = torch.cat(z_list, dim=0)
        z_splits[split_name] = z_all
        print(f"  [{split_name}] {z_all.shape[0]} molecules → z {z_all.shape}")

    # ── Compute properties ──
    print("\n[props] Computing molecular properties (QED, SA, LogP, MolWt)...")
    props = {}
    for split_name in ["train", "val", "test"]:
        split_props = compute_mol_properties(splits_smiles[split_name])
        valid_mask = ~torch.isnan(split_props).any(dim=-1)
        print(f"  [{split_name}] {valid_mask.sum().item()}/{len(splits_smiles[split_name])} valid")
        props[split_name] = split_props

    prop_names = ["qed", "sa_score", "logp", "molwt"]

    # ── Save ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "train": z_splits["train"],
        "val": z_splits["val"],
        "test": z_splits["test"],
        "latent_dim": vae_cfg.latent_dim,
        "vae_ckpt": str(args.vae_ckpt),
        "train_props": props["train"],
        "val_props": props["val"],
        "test_props": props["test"],
        "prop_names": prop_names,
        # SELFIES-specific metadata
        "vocab": vocab,
        "max_len": max_len,
        "train_smiles": splits_smiles["train"],
        "val_smiles": splits_smiles["val"],
        "test_smiles": splits_smiles["test"],
    }, out_path)
    print(f"\n[done] Saved to {out_path}")
    print(f"  train: z={z_splits['train'].shape}, props={props['train'].shape}")
    print(f"  val:   z={z_splits['val'].shape}, props={props['val'].shape}")
    print(f"  test:  z={z_splits['test'].shape}, props={props['test'].shape}")


if __name__ == "__main__":
    main()
