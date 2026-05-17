# Drifting Faithfulness Execution Checklist

This checklist tracks reviewer-facing faithful Drifting reproduction runs.
They are supplemental and separate from the stable 8-page AAAI draft.

## Current Gate Summary

- Strict faithful core: 4/4 complete.
- Allocation sweeps: 6/6 complete.
- The faithful-reproduction package is complete for the strict core
  and allocation sweep; promote it only as conservative supplemental
  evidence because target tracking remains modest.
- Allocation sweeps are secondary mechanism evidence: they improve
  diversity more reliably than QED target tracking.

## Run Matrix

| Group | Experiment | Status | Command |
|---|---|---|---|
| faithful_allocation | `rf_FD_ALLOC_NEG16_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/allocation/rf_FD_ALLOC_NEG16_QED_s42.yaml` |
| faithful_allocation | `rf_FD_ALLOC_NEG32_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/allocation/rf_FD_ALLOC_NEG32_QED_s42.yaml` |
| faithful_allocation | `rf_FD_ALLOC_POS01_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/allocation/rf_FD_ALLOC_POS01_QED_s42.yaml` |
| faithful_allocation | `rf_FD_ALLOC_POS16_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/allocation/rf_FD_ALLOC_POS16_QED_s42.yaml` |
| faithful_allocation | `rf_FD_ALLOC_POS32_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/allocation/rf_FD_ALLOC_POS32_QED_s42.yaml` |
| faithful_allocation | `rf_FD_ALLOC_POS64_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/allocation/rf_FD_ALLOC_POS64_QED_s42.yaml` |
| faithful_core | `rf_FD_STRICT_PLAIN_PHI_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/core/rf_FD_STRICT_PLAIN_PHI_QED_s42.yaml` |
| faithful_core | `rf_FD_STRICT_PROP_PHI_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/core/rf_FD_STRICT_PROP_PHI_QED_s42.yaml` |
| faithful_core | `rf_FD_STRICT_RANDOM_PHI_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/core/rf_FD_STRICT_RANDOM_PHI_QED_s42.yaml` |
| faithful_core | `rf_FD_STRICT_ZSPACE_QED_s42` | complete_pass | `python -m src.train.train_selfies_cfg --config configs/reviewer_faithful/core/rf_FD_STRICT_ZSPACE_QED_s42.yaml` |

## Automation And Audit Commands

Deferred core launcher:

```bash
python scripts/defer_faithful_core_after_destructive.py \
  --watch-status outputs/publication_ext/parallel_runner_status.json \
  --faithful-status outputs/reviewer_faithful/core_status.json \
  --devices 0,2,3 \
  --poll-seconds 60 \
  --log-dir outputs/reviewer_faithful/logs \
  --pid-file outputs/reviewer_faithful/deferred_faithful_core_launcher.pid
```

Completion audit:

```bash
python scripts/collect_faithful_drifting_results.py
python scripts/render_faithful_supplement.py
TEXINPUTS=docs//: BSTINPUTS=docs//: latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=DriftingMol_AAAI_FaithfulSupplement docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex
python scripts/audit_drifting_faithfulness.py
python scripts/audit_reviewer_experiment_readiness.py
```

Completion gate: the strict core and allocation rows are complete.
The faithful-reproduction claim may be described as completed
supplemental evidence, with the conservative interpretation stated
above.
