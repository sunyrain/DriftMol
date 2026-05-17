"""Discrete graph → SMILES conversion with optional valence repair / constrained decoding.

Given argmax-decoded (or constrained-decoded) node/edge tensors, reconstructs
RDKit Mol objects and attempts sanitization.

Three decoding modes:
  1. **argmax** (default): independent argmax per edge — fast but may violate valence.
  2. **repair** (``repair_overvalence=True``): argmax + iterative bond removal.
  3. **constrained** (``constrained_decode_edge_logits``): greedy assignment respecting
     valence limits from the start — guarantees zero valence violations, no post-hoc fix.

Key exports:
  - ``graph_tensors_to_mol`` / ``mol_to_smiles`` – single-graph conversion
  - ``graphs_to_valid_smiles`` – batch conversion, returns list of valid SMILES
  - ``constrained_decode_edge_logits`` – valence-constrained edge assignment from logits
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from src.data.featurizer import BOND_INDEX_TO_TYPE

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "RDKit is required for chemistry validity checks. "
        "Install with `conda install -c conda-forge rdkit`."
    ) from exc


@dataclass
class DecodeConfig:
    atom_index_to_number: dict[int, int]
    repair_overvalence: bool = False
    max_repair_steps: int = 32


_ATOMIC_MAX_VALENCE = {
    1: 1.0,   # H
    6: 4.0,   # C
    7: 3.0,   # N
    8: 2.0,   # O
    9: 1.0,   # F
}

_BOND_CLASS_ORDER = {
    1: 1.0,   # single
    2: 2.0,   # double
    3: 3.0,   # triple
}


def _repair_edge_types(
    node_type: torch.Tensor,
    edge_type: torch.Tensor,
    decode_cfg: DecodeConfig,
) -> torch.Tensor:
    repaired = edge_type.clone()
    node_ids = [i for i, t in enumerate(node_type.tolist()) if t != 0]
    if not node_ids:
        return repaired

    max_vals = {}
    for idx in node_ids:
        atom_cls = int(node_type[idx].item())
        atomic_num = decode_cfg.atom_index_to_number.get(atom_cls)
        max_vals[idx] = _ATOMIC_MAX_VALENCE.get(atomic_num, 4.0)

    for _ in range(max(0, int(decode_cfg.max_repair_steps))):
        over_atom = None
        over_amount = 0.0
        over_neighbors: list[tuple[int, int, float]] = []

        for i in node_ids:
            cur_val = 0.0
            neighbors = []
            for j in node_ids:
                if j == i:
                    continue
                bcls = int(repaired[i, j].item())
                if bcls == 0:
                    continue
                bval = _BOND_CLASS_ORDER.get(bcls, 1.0)
                cur_val += bval
                neighbors.append((i, j, bval))

            exceed = cur_val - max_vals[i]
            if exceed > over_amount:
                over_amount = exceed
                over_atom = i
                over_neighbors = neighbors

        if over_atom is None or over_amount <= 1e-6 or not over_neighbors:
            break

        # Remove the heaviest bond attached to the most over-valent atom
        _, j_remove, _ = max(over_neighbors, key=lambda x: x[2])
        repaired[over_atom, j_remove] = 0
        repaired[j_remove, over_atom] = 0

    return repaired


def constrained_decode_edge_logits(
    node_type: torch.Tensor,
    edge_logits: torch.Tensor,
    atom_index_to_number: dict[int, int],
) -> torch.Tensor:
    """Greedy valence-constrained edge assignment from logits.

    Instead of independent argmax (which may violate valence), this assigns
    bonds greedily in order of model confidence, respecting each atom's
    remaining valence budget at every step.

    Args:
        node_type: [N] long atom class indices (0=PAD, 1=H, 2=C, …)
        edge_logits: [N, N, 4] float logits for bond classes (0=none,1=single,2=double,3=triple)
        atom_index_to_number: {class_idx: atomic_number} mapping

    Returns:
        edge_type: [N, N] long tensor of assigned bond classes.
                   Guaranteed to respect valence for all atoms.
    """
    node_type = node_type.detach().cpu().long()
    edge_logits = edge_logits.detach().cpu().float()
    N = node_type.shape[0]
    edge_type = torch.zeros(N, N, dtype=torch.long)

    # Identify real atoms and their max valence
    real_atoms: list[int] = []
    remaining_valence: dict[int, float] = {}
    for i in range(N):
        cls = int(node_type[i].item())
        if cls == 0:  # PAD
            continue
        real_atoms.append(i)
        atomic_num = atom_index_to_number.get(cls)
        remaining_valence[i] = _ATOMIC_MAX_VALENCE.get(atomic_num, 4.0) if atomic_num else 0.0

    if len(real_atoms) < 2:
        return edge_type

    # Compute bond probabilities (symmetrize upper/lower triangle)
    bond_probs = torch.softmax(edge_logits, dim=-1)  # [N, N, 4]

    # Collect candidate edges: (i, j, desire_score, symmetrized_probs)
    candidates: list[tuple[int, int, float, torch.Tensor]] = []
    for idx_a in range(len(real_atoms)):
        i = real_atoms[idx_a]
        for idx_b in range(idx_a + 1, len(real_atoms)):
            j = real_atoms[idx_b]
            # Symmetrize: average probs from both triangle directions
            avg_p = (bond_probs[i, j] + bond_probs[j, i]) / 2.0
            # Desire = max probability for any non-zero bond type
            desire = avg_p[1:].max().item()
            candidates.append((i, j, desire, avg_p))

    # Sort by desire (most confident bonds first → greedy priority)
    candidates.sort(key=lambda x: -x[2])

    # Greedy assignment
    for i, j, desire, probs in candidates:
        # Sort bond types by probability (descending)
        sorted_types = probs.argsort(descending=True)

        assigned = 0
        for bond_cls_t in sorted_types:
            k = int(bond_cls_t.item())
            if k == 0:
                # Model prefers no-bond over all remaining types → assign no bond
                break
            if k <= remaining_valence.get(i, 0) and k <= remaining_valence.get(j, 0):
                assigned = k
                break

        edge_type[i, j] = assigned
        edge_type[j, i] = assigned
        if assigned > 0:
            remaining_valence[i] -= assigned
            remaining_valence[j] -= assigned

    return edge_type


def constrained_decode_batch(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    atom_index_to_number: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch constrained decoding: node argmax + valence-constrained edges.

    Args:
        node_logits: [B, N, C_node] float logits
        edge_logits: [B, N, N, C_edge] float logits
        atom_index_to_number: mapping

    Returns:
        node_types: [B, N] long, edge_types: [B, N, N] long
    """
    B = node_logits.shape[0]
    node_types = node_logits.argmax(dim=-1)  # [B, N]
    edge_types_list = []
    for b in range(B):
        et = constrained_decode_edge_logits(
            node_types[b], edge_logits[b], atom_index_to_number
        )
        edge_types_list.append(et)
    return node_types, torch.stack(edge_types_list)


def graph_tensors_to_mol(
    node_type: torch.Tensor,
    edge_type: torch.Tensor,
    decode_cfg: DecodeConfig,
) -> Optional[Chem.Mol]:
    """
    Convert fixed-size graph tensors to RDKit Mol.

    Args:
        node_type: [N] long class ids (0 is PAD)
        edge_type: [N, N] long class ids (0 is no-bond)
    """
    node_type = node_type.detach().cpu().long()
    edge_type = edge_type.detach().cpu().long()

    if decode_cfg.repair_overvalence:
        edge_type = _repair_edge_types(node_type=node_type, edge_type=edge_type, decode_cfg=decode_cfg)

    node_ids = [i for i, t in enumerate(node_type.tolist()) if t != 0]
    if not node_ids:
        return None

    rw_mol = Chem.RWMol()
    old_to_new: dict[int, int] = {}

    for old_idx in node_ids:
        atom_cls = int(node_type[old_idx].item())
        atomic_num = decode_cfg.atom_index_to_number.get(atom_cls)
        if atomic_num is None:
            return None
        atom = Chem.Atom(atomic_num)
        new_idx = rw_mol.AddAtom(atom)
        old_to_new[old_idx] = new_idx

    for i in node_ids:
        for j in node_ids:
            if j <= i:
                continue
            bond_cls = int(edge_type[i, j].item())
            if bond_cls == 0:
                continue
            bond_type = BOND_INDEX_TO_TYPE.get(bond_cls)
            if bond_type is None:
                return None
            rw_mol.AddBond(old_to_new[i], old_to_new[j], bond_type)

    try:
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def mol_to_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def graphs_to_valid_smiles(
    graphs: Iterable[dict[str, torch.Tensor]],
    decode_cfg: DecodeConfig,
) -> list[str]:
    valid_smiles: list[str] = []
    for graph in graphs:
        mol = graph_tensors_to_mol(
            node_type=graph["node_type"],
            edge_type=graph["edge_type"],
            decode_cfg=decode_cfg,
        )
        if mol is not None:
            valid_smiles.append(mol_to_smiles(mol))
    return valid_smiles

