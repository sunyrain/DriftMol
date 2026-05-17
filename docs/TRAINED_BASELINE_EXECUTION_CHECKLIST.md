# Trained Baseline Execution Checklist

This checklist tracks additional reviewer-facing trained baselines that are
separate from the stable 8-page AAAI package and from the faithful-Drifting
reproduction queue.

## Completed Runs

| Group | Experiment | Status | Purpose | Command |
|---|---|---|---|---|
| trained_baseline | `ext_B_LINEAR_PROP_QED_s42` | complete/pass | Fixed linear latent-property guidance baseline with no drift and no QED binning. Tests whether a simple property regressor can replace drift under the same latent generator backbone. | `python -m src.train.train_selfies_cfg --config configs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s42.yaml` |
| trained_baseline | `ext_B_LINEAR_PROP_QED_s43` | complete/pass | Second seed for the same baseline, added to make the comparison reviewer-usable rather than single-seed. | `python -m src.train.train_selfies_cfg --config configs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s43.yaml` |
| trained_baseline | `ext_B_LINEAR_PROP_QED_s44` | complete/pass | Third seed for the same baseline. | `python -m src.train.train_selfies_cfg --config configs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s44.yaml` |

## Artifacts

- Configs: `configs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s4{2,3,4}.yaml`
- Manifest: `configs/publication_ext/baseline_manifest.json`
- Runner status: `outputs/publication_ext/parallel_runner_status_baseline*.json`
- Runner logs: `outputs/publication_ext/baseline_logs/ext_B_LINEAR_PROP_QED_s4*.log`
- PID files: `outputs/publication_ext/baseline_runner*.pid`
- Seed-44 watcher log: `outputs/publication_ext/baseline_runner_s44_watcher.log`
- Seed-44 watcher PID: `outputs/publication_ext/baseline_runner_s44_watcher.pid`
- Postprocess watcher log: `outputs/publication_ext/baseline_postprocess.log`
- Postprocess watcher PID: `outputs/publication_ext/baseline_postprocess.pid`
- Collector: `scripts/collect_trained_baselines.py`
- Result CSV: `results/trained_baseline_qed.csv`
- Status JSON: `results/trained_baseline_status.json`
- LaTeX table: `results/tables/tab_trained_baseline_qed.tex`

## Launch Command

```bash
setsid python scripts/run_manifest_parallel.py \
  --manifest configs/publication_ext/baseline_manifest.json \
  --group trained_baseline \
  --devices 2 \
  --poll-seconds 30 \
  --status-file outputs/publication_ext/parallel_runner_status_baseline.json \
  --log-dir outputs/publication_ext/baseline_logs \
  > outputs/publication_ext/baseline_runner.log 2>&1 < /dev/null &
```

Second-seed launch on GPU0:

```bash
setsid python scripts/run_manifest_parallel.py \
  --manifest configs/publication_ext/baseline_manifest.json \
  --name ext_B_LINEAR_PROP_QED_s43 \
  --devices 0 \
  --poll-seconds 30 \
  --status-file outputs/publication_ext/parallel_runner_status_baseline_s43.json \
  --log-dir outputs/publication_ext/baseline_logs \
  > outputs/publication_ext/baseline_runner_s43.log 2>&1 < /dev/null &
```

Deferred third-seed watcher:

```bash
setsid bash -c 'while true; do
  if [ -f outputs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s42/final_metrics.json ]; then
    python scripts/run_manifest_parallel.py \
      --manifest configs/publication_ext/baseline_manifest.json \
      --name ext_B_LINEAR_PROP_QED_s44 \
      --devices 2 \
      --poll-seconds 30 \
      --status-file outputs/publication_ext/parallel_runner_status_baseline_s44.json \
      --log-dir outputs/publication_ext/baseline_logs
    exit $?
  fi
  sleep 60
done' > outputs/publication_ext/baseline_runner_s44_watcher.log 2>&1 < /dev/null &
```

Postprocess watcher:

```bash
setsid bash -c 'while true; do
  count=$(find outputs/publication_ext/baselines -mindepth 2 -maxdepth 2 -name final_metrics.json | wc -l)
  if [ "$count" -ge 3 ]; then break; fi
  sleep 300
done
python scripts/collect_trained_baselines.py
python scripts/audit_reviewer_experiment_readiness.py || true' \
  > outputs/publication_ext/baseline_postprocess.log 2>&1 < /dev/null &
```

## Result And Claim Rule

All three registered seeds are complete. The publishable summary is
`results/tables/tab_trained_baseline_qed.tex`: mean QED rho `0.046 +/- 0.150`,
uniqueness `92.9% +/- 2.6%`, MAE `0.335 +/- 0.096`, and slope
`0.081 +/- 0.202`. This supports the claim that plain fixed property-regression
guidance does not replace DriftingMol's drift field.

Refresh artifacts with:

```bash
python scripts/collect_trained_baselines.py
```
