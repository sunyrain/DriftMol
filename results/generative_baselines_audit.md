# Generative Baseline Audit

| Requirement | Evidence | Status |
|---|---|---|
| Manifest exists and has 12 entries | configs/publication_ext/generative_baselines_manifest.json entries=12 | PASS |
| Manifest covers four families and three seeds | methods=['cvae', 'diffusion', 'flow_matching', 'gan'], seeds=[42, 43, 44] | PASS |
| Every manifest output has final_metrics.json | all present | PASS |
| Summary JSON has 12 result rows | results/generative_baselines_qed.json rows=12 | PASS |
| Aggregates report n=3 with core metrics for each family | cvae n=3, gan n=3, diffusion n=3, flow_matching n=3 | PASS |
| LaTeX table exists and includes all four families | results/tables/tab_generative_baselines_qed.tex | PASS |

Overall: PASS
