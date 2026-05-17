# Graph Namespace Adapter Plan

Updated: 2026-05-15 UTC

Status: implemented on 2026-05-15 UTC.

This plan closed the non-GPU part of the graph-stress launch blocker. The
archived graph route imports `src.*`, while the active repository now uses
`src` for the SELFIES line. Fresh graph runs therefore execute with an
archive-local `src` package before any GPU training is launched.

## Required Mapping

Implemented `archive/graph_vae_line/src/` as an archive-local package. It
resolves imports as follows:

| Archived import | Archive-local target |
|---|---|
| `src.data.featurizer` | `archive/graph_vae_line/src_data/featurizer.py` |
| `src.data.qm9_dataset` | `archive/graph_vae_line/src_data/qm9_dataset.py` |
| `src.drifting.drift_v2` | `archive/graph_vae_line/src_drifting/drift_v2.py` |
| `src.eval.metrics_molecular` | `archive/graph_vae_line/src_eval/metrics_molecular.py` |
| `src.eval.soft_valence` | `archive/graph_vae_line/src_eval/soft_valence.py` |
| `src.eval.validity_rdkit` | `archive/graph_vae_line/src_eval/validity_rdkit.py` |
| `src.models.graph_transformer_ae` | `archive/graph_vae_line/src_models/graph_transformer_ae.py` |
| `src.models.graph_transformer_ae_v2` | `archive/graph_vae_line/src_models/graph_transformer_ae_v2.py` |
| `src.train.train_generator` | `archive/graph_vae_line/src_train/train_generator.py` |
| `src.train.train_vae` | `archive/graph_vae_line/src_train/train_vae.py` |

The adapter must also provide `src.models.latent_mae` and `src.utils`. The
former can load the current repository's `src/models/latent_mae.py` by file
path, because the archived graph route used the same latent-MAE class name but
does not contain a local copy. The latter must expose graph-compatible
functions:

- `load_config`
- `set_seed`
- `build_lr_scheduler`
- `LatentDiTGenerator`
- `LatentDiTGeneratorCFG`
- `build_latent_generator`
- `load_vae`
- `discretize_logits`

`load_vae` should instantiate the graph VAE class from the checkpoint
configuration when available and load the `model_state_dict` / `model` payload
without touching the current SELFIES VAE code. `discretize_logits` can reuse
the graph implementation in `src_train/train_vae.py`.

## Smoke Tests Before GPU Launch

Run these from the archive directory after creating the adapter:

```bash
cd archive/graph_vae_line
PYTHONPATH=. python - <<'PY'
from src.data.qm9_dataset import QM9GraphDataset
from src.eval.validity_rdkit import DecodeConfig
from src.models.graph_transformer_ae import GraphTransformerAE
from src.models.graph_transformer_ae_v2 import GraphTransformerAE_V2
from src.models.latent_mae import LatentMAE
from src.train.train_generator import sample_and_evaluate
from src.train.train_vae import discretize_logits
from src.utils import build_latent_generator, build_lr_scheduler, load_config, load_vae
print("graph namespace smoke import: OK")
PY
```

Then rerun:

```bash
python ../../scripts/audit_graph_archive_launchability.py
```

Expected outcome before checkpoint recovery: namespace blockers should be
cleared, while missing checkpoint/cache blockers should remain OPEN. Only after
`outputs/vae_v3_valence/best.pt`,
`data/cache/qm9_latent_cache_v3.pt`, and
`outputs/latent_mae_v3/best_latent_mae.pt` exist should the graph-control rows
be launched.

Observed on 2026-05-15 UTC: the smoke import passed, `load_vae` reconstructed a
dummy graph checkpoint, the current `src.utils` compatibility layer now exposes
`load_vae` and `discretize_logits`, and the namespace blockers were cleared.
The audit remains OPEN only because the graph checkpoints/cache are still
absent.

## Claim Boundary

This adapter is an execution precondition, not evidence of graph performance.
Archived E36/E40 metrics remain diagnostic until fresh graph QED/LogP control,
graph destructive ablation, and raw-vs-repaired decoding diagnostics produce
new final metrics.
