"""Build a simple {smiles: [...]} cache for ZINC250K (same format as qm9_graph_cache.pt)."""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rdkit import Chem

raw = pathlib.Path("data/raw/zinc250k_clean.smi")
smiles = [l.strip() for l in open(raw) if l.strip()]
print(f"Raw SMILES: {len(smiles)}")

# Canonicalize
canonical = []
for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        canonical.append(Chem.MolToSmiles(mol))
print(f"Canonical valid: {len(canonical)}")

out = pathlib.Path("data/cache/zinc250k_smiles_cache.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"smiles": canonical}, out)
print(f"Saved to {out}")
