# Next-Wave Reviewer Experiment Plan

Updated: 2026-05-15 UTC

This plan records valuable follow-up experiments that were launched
opportunistically and are now complete.

## Resource Decision

Current state:

- Next-wave rows are 4/4 complete/pass in `results/next_wave_status.json`.
- Reviewer-extra rows are 4/4 complete/pass in `results/reviewer_extra_status.json`.
- `/root/autodl-tmp` has about 13G free after checkpoint cleanup.
- The 20G line is a conservative preflight target, not a hard training limit;
  with dynamic `last.pt` cleanup enabled, launches may proceed below that
  line if the active queue is managed carefully.

Decision outcome: the property-guidance baseline group and the second
continuous-QED seed were launched, completed, collected, and audited.

## Added Experiments

| Experiment | Purpose | Why it matters |
|---|---|---|
| LogP linear property-guidance baseline | Fixed Ridge head, no Drifting, LogP target | Compares LogP transfer against a direct non-drift baseline |
| SA-score linear property-guidance baseline | Fixed Ridge head, no Drifting, SA target | Checks whether SA transfer needs the drift field |
| Multi-property linear guidance baseline | Fixed Ridge head, no Drifting, four targets | Tests whether multi-property no-binning control needs Drifting |
| Continuous QED control, seed 43 | Second seed for no-binning QED | Tests whether the continuous-conditioning conclusion is stable |

Artifacts:

- Config generator: `scripts/generate_next_wave_configs.py`
- Manifest: `configs/publication_ext/next_wave_manifest.json`
- Collector: `scripts/collect_next_wave_results.py`
- Status: `results/next_wave_status.json`
- Table skeleton: `results/tables/tab_next_wave.tex`

## Launch Rules Used

Launch rows only under these rules:

1. Run fixed-guidance property baselines when their completed comparator rows
   already exist and a GPU is free enough for the run.
2. Keep the continuous-QED seed 43 row deferred until the seed-42 continuous
   comparator is complete.
3. Keep disk cleanup available to avoid filesystem pressure;
   the 20G line is advisory when the queue is already using dynamic cleanup.

## Launch Pattern Used

The GPU0 launch used:

```bash
python scripts/run_manifest_parallel.py \
  --manifest configs/publication_ext/next_wave_manifest.json \
  --group property_guidance_baseline \
  --devices 0 \
  --poll-seconds 30 \
  --status-file outputs/publication_ext/parallel_runner_status_next_wave_gpu0.json \
  --log-dir outputs/publication_ext/next_wave_logs
```

Then refresh:

```bash
python scripts/collect_next_wave_results.py
python scripts/audit_extension_completion.py || true
```

These rows are now inside the completed reviewer evidence pack. They remain
additional reviewer ammunition rather than the central paper claim.
