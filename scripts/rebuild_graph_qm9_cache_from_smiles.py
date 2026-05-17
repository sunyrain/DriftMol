#!/usr/bin/env python3
"""Rebuild the archived graph-route QM9 cache from the recovered SMILES list."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "graph_vae_line"
SOURCE_SMILES_CACHE = ROOT / "archive" / "data_qm9" / "qm9_graph_cache.pt"
OUT_SMILES = ARCHIVE / "data" / "qm9" / "qm9_from_cache.smi"
OUT_GRAPH_CACHE = ARCHIVE / "data" / "cache" / "qm9_graph_cache.pt"


def _load_smiles(path: Path) -> list[str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    smiles = payload.get("smiles") if isinstance(payload, dict) else None
    if not isinstance(smiles, list) or not smiles:
        raise RuntimeError(f"{path} does not contain a non-empty 'smiles' list")
    return [str(item).strip() for item in smiles if str(item).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create archive/graph_vae_line graph dataset files from recovered QM9 SMILES."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_SMILES_CACHE)
    parser.add_argument("--smiles-out", type=Path, default=OUT_SMILES)
    parser.add_argument("--cache-out", type=Path, default=OUT_GRAPH_CACHE)
    parser.add_argument("--force", action="store_true", help="Rebuild graph cache even if it already exists.")
    parser.add_argument("--max-nodes", type=int, default=29)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)

    args.smiles_out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_out.parent.mkdir(parents=True, exist_ok=True)

    smiles = _load_smiles(args.source)
    args.smiles_out.write_text("\n".join(smiles) + "\n", encoding="utf-8")
    print(f"[smiles] wrote {len(smiles)} rows to {args.smiles_out}")

    if args.force and args.cache_out.exists():
        args.cache_out.unlink()

    sys.path.insert(0, str(ARCHIVE))
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass
    from src.data.qm9_dataset import QM9GraphDataset

    ds = QM9GraphDataset(
        data_path=str(args.smiles_out.resolve()),
        split="train",
        split_seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        max_nodes=args.max_nodes,
        atom_types=(1, 6, 7, 8, 9),
        remove_hs=True,
        cache_path=str(args.cache_out.resolve()),
    )
    payload = torch.load(args.cache_out, map_location="cpu", weights_only=False)
    graphs = payload.get("graphs", [])
    cached_smiles = payload.get("smiles", [])
    print(f"[graph-cache] graphs={len(graphs)} smiles={len(cached_smiles)} path={args.cache_out}")
    print(f"[split-check] train={len(ds)} cache_meta_data_path={payload.get('meta', {}).get('data_path')}")


if __name__ == "__main__":
    main()
