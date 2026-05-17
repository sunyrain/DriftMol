"""Molecular generation evaluation metrics.

Computes standard unconditional generation metrics:
  - **Validity**: fraction of generated graphs that yield a valid SMILES.
  - **Uniqueness**: fraction of unique canonical SMILES among valid molecules.
  - **Novelty**: fraction of unique SMILES not present in the training set.

Also reports auxiliary statistics (mean atom/bond counts).
"""
from __future__ import annotations

from typing import Optional

import torch

from .validity_rdkit import DecodeConfig, graphs_to_valid_smiles


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def evaluate_generated_graphs(
    generated_graphs: list[dict[str, torch.Tensor]],
    decode_cfg: DecodeConfig,
    train_smiles: Optional[set[str]] = None,
) -> dict[str, float]:
    """
    Compute basic molecular generation metrics.

    Metrics:
    - validity
    - uniqueness among valid molecules
    - novelty against train set (if provided)
    - mean atom count (all generated samples)
    - mean bond count (all generated samples)
    """
    num_samples = len(generated_graphs)
    valid_smiles = graphs_to_valid_smiles(generated_graphs, decode_cfg)
    num_valid = len(valid_smiles)
    unique_smiles = set(valid_smiles)
    num_unique = len(unique_smiles)

    if train_smiles is not None and num_unique > 0:
        novel = [s for s in unique_smiles if s not in train_smiles]
        novelty = _safe_div(len(novel), num_unique)
    else:
        novelty = 0.0

    atom_counts = []
    bond_counts = []
    for graph in generated_graphs:
        node_type = graph["node_type"]
        edge_type = graph["edge_type"]
        atoms = int((node_type != 0).sum().item())
        # For undirected adjacency matrices count upper triangular bonds.
        bonds = int(torch.triu((edge_type != 0).long(), diagonal=1).sum().item())
        atom_counts.append(atoms)
        bond_counts.append(bonds)

    metrics = {
        "num_samples": float(num_samples),
        "num_valid": float(num_valid),
        "validity": _safe_div(num_valid, num_samples),
        "num_unique_valid": float(num_unique),
        "uniqueness": _safe_div(num_unique, num_valid),
        "novelty": novelty,
        "mean_num_atoms": float(sum(atom_counts) / max(1, len(atom_counts))),
        "mean_num_bonds": float(sum(bond_counts) / max(1, len(bond_counts))),
    }
    return metrics

