"""Molecular featurization: SMILES/SDF → fixed-size graph tensors.

Converts RDKit Mol objects into padded tensor representations suitable for
graph neural networks.  Each molecule is represented as:
  - **node_type** (N,)  – atom class index (0=PAD, 1=H, 2=C, 3=N, 4=O, 5=F)
  - **edge_type** (N, N) – bond class index (0=none, 1=single, 2=double, 3=triple)
  - **node_mask** (N,)   – 1 for real atoms, 0 for padding

Aromatic bonds are Kekulized (converted to alternating single/double) before
featurization, following MoFlow / GraphMVP convention.  This eliminates the
need for the model to learn valid aromatic ring placement.

Also provides helpers to read SMILES from files and canonicalize them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "RDKit is required for molecular featurization. "
        "Install with `conda install -c conda-forge rdkit`."
    ) from exc


# Bond class 0 is reserved for "no bond".
# Only 4 classes: no-bond(0), single(1), double(2), triple(3).
# Aromatic bonds are Kekulized before featurization.
BOND_INDEX_TO_TYPE = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}
BOND_TYPE_TO_INDEX = {v: k for k, v in BOND_INDEX_TO_TYPE.items()}


@dataclass(frozen=True)
class MoleculeFeaturizerConfig:
    max_nodes: int = 29
    atom_types: tuple[int, ...] = (1, 6, 7, 8, 9)
    remove_hs: bool = True

    @property
    def atom_index_to_number(self) -> Dict[int, int]:
        # Atom class 0 is reserved for PAD.
        return {idx + 1: atomic_num for idx, atomic_num in enumerate(self.atom_types)}

    @property
    def atom_number_to_index(self) -> Dict[int, int]:
        return {atomic_num: idx for idx, atomic_num in self.atom_index_to_number.items()}

    @property
    def num_atom_classes(self) -> int:
        return len(self.atom_types) + 1

    @property
    def num_bond_classes(self) -> int:
        return len(BOND_INDEX_TO_TYPE) + 1


def canonicalize_mol(mol: Chem.Mol, remove_hs: bool = True) -> Optional[Chem.Mol]:
    """Return a sanitized canonical molecule object or None if conversion fails."""
    if mol is None:
        return None
    try:
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    """Return canonical SMILES for a sanitized molecule."""
    return Chem.MolToSmiles(mol, canonical=True)


def mol_to_graph_tensors(
    mol: Chem.Mol,
    config: MoleculeFeaturizerConfig,
) -> Optional[dict[str, torch.Tensor]]:
    """
    Convert RDKit Mol into fixed-size graph tensors.

    Returns:
        dict with:
            node_type: LongTensor [N]
            edge_type: LongTensor [N, N]
            node_mask: BoolTensor [N]
        or None if molecule is unsupported by config.
    """
    mol = canonicalize_mol(mol, remove_hs=config.remove_hs)
    if mol is None:
        return None

    # Kekulize: convert aromatic bonds to alternating single/double
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0 or num_atoms > config.max_nodes:
        return None

    node_type = torch.zeros(config.max_nodes, dtype=torch.long)
    node_mask = torch.zeros(config.max_nodes, dtype=torch.bool)
    edge_type = torch.zeros((config.max_nodes, config.max_nodes), dtype=torch.long)

    atom_map = config.atom_number_to_index
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        atom_cls = atom_map.get(atomic_num)
        if atom_cls is None:
            return None
        node_type[atom_idx] = atom_cls
        node_mask[atom_idx] = True

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_cls = BOND_TYPE_TO_INDEX.get(bond.GetBondType())
        if bond_cls is None:
            return None
        edge_type[i, j] = bond_cls
        edge_type[j, i] = bond_cls

    return {
        "node_type": node_type,
        "edge_type": edge_type,
        "node_mask": node_mask,
    }


def graph_num_nodes(node_mask: torch.Tensor) -> int:
    """Count real nodes in a fixed-size graph tensor."""
    return int(node_mask.sum().item())


def iter_mols_from_file(path: str, smiles_column: str = "smiles") -> Iterable[Chem.Mol]:
    """
    Yield RDKit molecules from common molecular file formats.

    Supported:
        - .sdf
        - .smi / .smiles / .txt  (SMILES per line; first token is used)
        - .csv (expects smiles column name, default: "smiles")
    """
    import csv
    from pathlib import Path

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(file_path), removeHs=False, sanitize=False)
        for mol in supplier:
            if mol is not None:
                yield mol
        return

    if suffix in {".smi", ".smiles", ".txt"}:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                smiles = text.split()[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    yield mol
        return

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if smiles_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV file {path} does not contain smiles column '{smiles_column}'. "
                    f"Available columns: {reader.fieldnames}"
                )
            for row in reader:
                smiles = (row.get(smiles_column) or "").strip()
                if not smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    yield mol
        return

    raise ValueError(
        f"Unsupported file format for {path}. "
        "Supported: .sdf, .smi, .smiles, .txt, .csv"
    )

