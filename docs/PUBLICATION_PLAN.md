# DriftingMol AAAI Comprehensive Plan

Updated: 2026-05-15 UTC

This is the active plan for turning the current DriftingMol AAAI draft into a
stronger submission. It separates the already-audited 8-page SELFIES manuscript
from the now-complete reviewer extension evidence pack.

## North Star

Submit DriftingMol as a focused paper about single-step property-conditional
molecular generation in a SELFIES latent space, with coupled decoder drift as
the mechanism. The paper should not become a full graph-generation paper.
Graph experiments should instead be used as representation stress tests that
explain why the main contribution is built on SELFIES.

The target reviewer reading is:

1. The method is clear and formally grounded.
2. The key mechanism is tested by destructive ablations.
3. The gains are not explained by trivial retrieval, local latent jitter, or a
   weak baseline protocol.
4. SELFIES validity is presented honestly as a representation guarantee.
5. Graph decoding limitations are acknowledged and quantified rather than
   ignored.
6. Figures look like a professional conference paper, not an internal report.

## Current State

The stable main package is complete:

- Main source: `docs/PAPER_AAAI.tex`
- Main PDF: `DriftingMol_AAAI.pdf`
- Current main PDF length: 8 pages
- Checklist source/PDF: `docs/AAAI_REPRODUCIBILITY_CHECKLIST.tex`,
  `DriftingMol_AAAI_Checklist.pdf`
- Submission bundle: `submission/driftingmol_aaai2026_submission_draft.zip`
- Main completion gate: `python scripts/audit_publication_completion.py --run-tests`
  reports PASS for the existing publication package.

The extension phase is complete under the current reviewer-facing gates:

- Destructive drift ablations are complete and collected.
- VAE sensitivity is complete for Low-beta, High-beta, Latent-128, and
  Decoder-6, and all four alternative-checkpoint downstream QED drifting rows
  have final metrics.
- Graph stress evidence includes the archived diagnostic snapshot and a closed
  fresh graph-route package with QED, LogP, no-drift, raw/repaired validity,
  and graph-vs-SELFIES comparison artifacts.
- Protocol-matched baselines exist for retrieval, VAE jitter, and bin-Gaussian;
  the fixed linear latent-property guidance baseline is complete across three
  seeds and is weak enough to support the drift-specific mechanism claim.
- Generalization rows for LogP, SA-score, and two multi-property seeds are
  complete; reviewer-extra robustness and next-wave property-guidance controls
  are also complete.
- Same-backbone conditional latent VAE, WGAN-GP, DDPM, and Flow-Matching
  generative baselines are complete across seeds 42, 43, and 44.
- The main conceptual figure uses a documented image-2 raster asset, and the
  quantitative figures have been regenerated with cleaner labels and checked
  against the 8-page PDF layout.

## Claim Boundary

The manuscript should make these claims:

- DriftingMol provides one-forward-pass conditional molecular generation in a
  frozen SELFIES VAE latent space.
- SELFIES decoding supplies validity by construction; validity is not claimed
  as a learned property of the drift objective.
- Coupled decoder drift is the useful mechanism: decoder hidden features define
  the feature space, and the decoder Jacobian pulls the drift gradient back to
  the latent generator.
- The method provides property-biased generation, not exact molecular
  optimization.

The manuscript should avoid these claims unless new evidence supports them:

- General graph generation state of the art.
- Universal superiority over external optimization systems such as LIMO, FREED,
  MOOD, DiGress, GDSS, or MoFlow under their original protocols.
- Exact target matching.
- Validity as a model-learning contribution independent of SELFIES.

## Reviewer-Risk Map

| Risk | Likely reviewer criticism | Required response |
|---|---|---|
| Baselines | The comparison is mostly ablation/internal controls. | Add protocol-matched trained baselines and clearly label retrieval as a memorization reference. |
| Representation | SELFIES makes validity trivial; graph route is missing. | Add graph representation stress test and discuss decoder bottlenecks. |
| VAE dependence | Results may rely on one lucky SELFIES VAE. | Run VAE beta, latent-size, and decoder-capacity sensitivity. |
| Drifting fidelity | The method may not be a faithful reproduction of Drifting Models. | Run the reviewer-faithful latent-MAE phi package in `docs/DRIFTING_FAITHFULNESS_PLAN.md`. |
| Mechanism | Decoder drift may be incidental. | Complete destructive attraction/repulsion/normalization ablations. |
| Figures | Current figures look crowded and informal. | Redesign main flow and result figures with publication-grade visual language. |
| Page budget | Extra experiments may overflow 8 pages. | Keep main text lean; move additional tables to supplement/checklist notes. |

## Work Package P0: Submission-Critical

### P0.1 Destructive Drifting Ablations

Goal: show that the anti-symmetric attraction/repulsion construction and
normalization choices matter.

Current status:

- Code hooks exist in `src/drifting/drift_latent_phi.py`.
- Training config hooks exist in `src/train/train_selfies_cfg.py`.
- Seven configs exist under `configs/publication_ext/destructive/`.
- A parallel runner is active for group `destructive_drift`.

Required variants:

| Variant | Purpose | Expected useful outcome |
|---|---|---|
| Attraction-only | Remove repulsive balance. | Collapse or weak target control. |
| Repulsion-only | Remove attractive reference pull. | Weak target control or poor calibration. |
| Broken attraction scale | Break anti-symmetric balance. | Instability or diversity/control tradeoff. |
| Broken repulsion scale | Break anti-symmetric balance. | Instability or diversity/control tradeoff. |
| Y-only normalization | Test partial normalization. | Intermediate result. |
| No cross normalization | Test kernel balance. | Severe control failure. |
| No row/column normalization | Test estimator stability. | Instability or collapse. |

Minimum success criterion:

- At least three destructive variants complete with interpretable negative
  evidence, and the main paper can state that preserving the coupled balanced
  drift structure is empirically necessary.

Artifacts:

- `results/destructive_ablation.csv`
- `results/destructive_ablation_status.json`
- `results/tables/tab_destructive_ablation.tex`
- `docs/EXTENSION_EXECUTION_CHECKLIST.md`

### P0.2 VAE Architecture Sensitivity

Goal: answer the user's concern directly: the current route assumes SELFIES
validity, but it should not assume one VAE architecture is uniquely responsible
for the result.

Current status:

- Four configs exist under `configs/publication_ext/vae_sensitivity/`.
- Low-beta is complete with exact reconstruction `96.9%` and prior VUN
  `0.988`.
- High-beta is complete with exact reconstruction `18.4%` and prior VUN
  `0.986`.
- Latent-128 is complete with exact reconstruction `94.3%` and prior VUN
  `0.992`.
- Decoder-6 is complete with exact reconstruction `56.5%` and prior VUN
  `0.980`.
- Downstream QED drifting is complete for all four alternative VAEs: low-beta
  rho `0.437`, high-beta `0.282`, latent-128 `0.421`, and decoder-6 `0.272`.

Required minimum:

| Variant | Question |
|---|---|
| Lower beta | Does a less-regularized latent space improve reconstruction but weaken prior control? |
| Higher beta | Does a smoother latent prior improve drift stability at the cost of reconstruction? |
| Latent 128 | Does a tighter bottleneck preserve the qualitative mechanism? |
| Deeper decoder | Does extra decoder capacity change the coupled-decoder effect? |

Minimum success criterion:

- At least one credible alternative VAE reaches acceptable prior quality and
  supports the same qualitative conclusion after a downstream drift run. This
  criterion is now met by the low-beta and latent-128 rows, while high-beta and
  decoder-6 serve as useful sensitivity boundaries.

Decision rule:

- If all alternatives fail prior quality, report the limitation as VAE
  representation sensitivity and do not overclaim architecture robustness.
- If one alternative succeeds, add a compact sensitivity table in the supplement
  or a short main-text sentence.

### P0.3 Graph Representation Stress Test

Goal: address the graph route without changing the paper into a graph method.

Current status:

- Archived graph metrics are summarized in `docs/GRAPH_STRESS_TEST.md`.
- The diagnostic figure exists at `docs/figures/fig_graph_bottleneck.pdf`.
- A prepared graph stress manifest exists at
  `configs/publication_ext/graph_stress_manifest.json`; it records the
  recovery sequence, fresh E36/E40 runs, and destructive / decode diagnostics.
- `results/graph_stress_full_status.json` marks the full package complete.

Required minimum:

| Run | Scope | Required metrics |
|---|---|---|
| Graph VAE prior | QM9 or archived clean graph line. | V/U/N/VUN. |
| Graph drift unconditional | Same graph VAE. | V/U/N/VUN. |
| Graph QED control | Same target grid where possible. | rho, slope, MAE, V/U/N. |
| Graph LogP control | Same target grid where possible. | rho, slope, MAE, V/U/N. |
| Bottleneck diagnostic | phi-space, soft-decoder, argmax-decoder gaps. | Quantified gap. |

Interpretation rule:

- If graph control is weak, present it as evidence that graph discrete decoding
  creates a stronger representation bottleneck than SELFIES.
- If graph control is unexpectedly strong, keep it as supplementary evidence
  and do not expand the title or abstract around graph generation.

### P0.4 Protocol-Matched Baselines

Goal: show that DriftingMol is not only better than weak internal controls.

Already complete:

| Baseline | QED rho | Novelty | Interpretation |
|---|---:|---:|---|
| Target-bin retrieval | 0.999 | 0.0% | Memorization upper reference. |
| VAE latent jitter | 0.974 | 13.0% | Local-neighborhood reference. |
| Conditional bin Gaussian | 0.126 | 100.0% | Simple latent prior baseline. |

Strong next baselines:

| Baseline | Why it matters | Priority |
|---|---|---|
| Conditional SELFIES VAE prior | Natural same-backbone conditional generative baseline. | Complete |
| Latent property-predictor guidance | Tests whether a property regressor can replace drift. | Complete |
| Conditional latent WGAN-GP | Representative GAN-family same-backbone generator. | Complete |
| Conditional latent DDPM | Representative diffusion-family same-backbone generator. | Complete |
| Conditional latent Flow Matching | Representative flow-matching same-backbone generator. | Complete |

Minimum success criterion:

- The paper has at least one trained nontrivial baseline in addition to
  retrieval/jitter/bin-Gaussian, or the limitation is acknowledged explicitly.

Current trained-baseline execution:

- `ext_B_LINEAR_PROP_QED_s42`, `s43`, and `s44` are complete.
- The three-seed mean QED rho is `0.046 +/- 0.150`, with no drift, no QED
  binning, and a fixed Ridge/linear latent property head. This is direct
  evidence that simple property-regression guidance does not replace drift.
- The same-backbone generator-family comparison is complete for conditional
  latent VAE, WGAN-GP, DDPM, and Flow Matching across seeds 42, 43, and 44;
  these are the correct quantitative generative-model baselines.

### P0.5 Figure Redesign

Goal: make the paper visually credible.

Figure plan:

| Figure | Action |
|---|---|
| Figure 1 main flow | Use the image-2 generated raster artwork for the conceptual pipeline; keep text minimal and readable. |
| Figure 2 QED ablation | Use the script-generated quantitative ablation ranking with descriptive labels and no internal run IDs. |
| Figure 3/5 diversity and seed plots | Use clean journal-style script-generated plotting with readable labels, few annotations, and consistent colors. |
| Graph bottleneck figure | Keep compact and diagnostic; avoid making it look like a second main contribution. |

Rules:

- Conceptual panels can use image-2 raster assets.
- Quantitative values must remain traceable to generated result tables; final
  publication data figures should be regenerated from scripts rather than
  image-2.
- Avoid dense titles inside panels; captions should carry the explanation.
- Use model names such as "DriftingMol", "Single-temperature", "No-diversity",
  "z-space drift", and "random features"; do not use internal labels such as
  A6/A8/F/G4 in the paper.

### P0.6 Drifting-Faithfulness Supplement

Goal: prove that the molecular implementation faithfully reproduces the
original Drifting Models algorithm before presenting decoder coupling as the
molecular adaptation.

Artifacts:

- `docs/DRIFTING_FAITHFULNESS_PLAN.md`
- `docs/DRIFTING_ALGORITHM_AUDIT.md`
- `docs/SUPPLEMENT_FAITHFUL_DRIFTING.tex`
- `docs/SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex`
- `docs/REVIEWER_EXPERIMENT_MATRIX.md`
- `scripts/generate_faithful_drifting_configs.py`
- `scripts/collect_faithful_drifting_results.py`
- `scripts/render_faithful_supplement.py`
- `scripts/defer_faithful_core_after_destructive.py`
- `scripts/audit_drifting_faithfulness.py`
- `configs/reviewer_faithful/manifest.json`
- `results/faithful_drifting.csv`
- `results/tables/tab_faithful_drifting_core.tex`
- `results/tables/tab_faithful_drifting_allocation.tex`
- `results/drifting_faithfulness_audit.md`
- `results/drifting_faithfulness_status.json`

Required evidence:

| Evidence | Status target |
|---|---|
| Algorithm-2 code audit | direct tensor-level equivalence test passes |
| Strict protocol config audit | reviewer-faithful configs pass semantic checks for Algorithm 2, Appendix-A.6 normalization, generated negatives, and no molecule-specific diversity/coupling add-ons |
| Strict latent-MAE phi run | complete; final metrics exist for `rf_FD_STRICT_PLAIN_PHI_QED_s42` |
| Feature-quality control | complete; property-aware phi and random-phi controls have final metrics |
| No-feature control | complete; z-space control has final metrics and is the strongest strict core row so far |
| Positive/negative allocation | complete; six-run Table-2-style sweep has final metrics and is summarized in the supplement |

The strict core package is complete. The conservative interpretation is that a
faithful molecular analogue is measurable but modest: z-space reaches rho
`0.127`, random phi `0.122`, property-aware phi `0.105`, and plain phi `0.055`.
This weakens any feature-extractor superiority claim and supports the main
paper framing that decoder coupling is the molecule-specific adaptation. The
allocation sweep should mostly live in the supplement; the main paper only
needs a compact paragraph separating faithful reproduction from the
decoder-coupled extension.

## Work Package P1: Strongly Recommended

### P1.1 Three-Seed Coverage for Baselines

Run three seeds for the strongest trained baseline and compare it with the
three-seed DriftingMol group. This protects against the criticism that only the
proposed method received seed-level treatment.

### P1.2 Multi-Property Baseline Extension

Apply the strongest matched baseline to QED/SA/LogP/MolWt. This protects the
multi-property result from looking like an internal-only ablation.

### P1.3 NFE Tradeoff

Compare one-step DriftingMol against a small latent diffusion or flow-matching
baseline at 1, 5, 10, and 50 NFE.

Required metrics:

- QED/LogP rho, MAE, V/U/N
- Latents/s and end-to-end molecules/s
- Same target grid as DriftingMol

## Work Package P2: Optional and High Risk

These should not block submission:

- Full ZINC-scale graph VAE or graph diffusion reruns.
- Full external SOTA reruns for DiGress, GDSS, MoFlow, LIMO, FREED, or MOOD.
- Large DiT scaling beyond the current model.
- New molecular feature pretraining beyond current LatentMAE variants.

## Execution Order

### Immediate: 2026-05-15

1. Keep 30-minute monitoring active while the four generalization rows run.
2. Refresh collectors and audits after each generalization or reviewer-extra
   row completes; next-wave work can proceed with dynamic checkpoint cleanup
   once a GPU frees cleanly and the queue stays below the filesystem limit.
3. Integrate the completed VAE architecture and downstream VAE-drift evidence
   into the supplement and claim-boundary text.
4. Polish the main manuscript language, figure captions, and table naming so
   the paper reads like a conference submission rather than a run report.
5. Prepare the graph route as a documented limitation plus recovery plan until
   the archived graph namespace/checkpoint blockers are resolved.

### Short Term: 2026-05-14 to 2026-05-20

1. Finish generalization and reviewer-extra queues, then decide which rows are
   strong enough for the main paper versus supplement only.
2. Keep the strict reviewer-faithful latent-MAE phi core and allocation sweep
   in the supplement; both are complete and conservative.
3. Use the completed trained baseline as a weak fixed-guidance control, not as
   a broad external-baseline comparison.
4. Resolve graph stress launchability only after current GPU pressure drops.
5. Preserve the image-2 main conceptual figure, keep numerical result figures
   script-generated from data, and perform only final layout/aesthetic checks
   before submission freeze.

### Mid Term: 2026-05-21 to 2026-06-12

1. Add next-wave LogP/SA/multi-property fixed-guidance baselines only if the
   current generalization results are strong enough to justify the extra cost.
2. Run graph recovery/stress experiments if namespace isolation and missing
   checkpoint/cache recovery are resolved.
3. Regenerate all quantitative figures and tables from scripts; documented
   image-2 assets are limited to the conceptual pipeline figure.
4. Rewrite experiments and limitations around the completed generalization,
   VAE-sensitivity, and graph-stress evidence.

### Final Development: 2026-06-13 to 2026-07-15

1. Run P1 NFE tradeoff and multi-property baseline only if P0 is already stable.
2. Perform reviewer-score simulation and revise claims toward score 8.
3. Freeze experiments by mid-July.
4. Spend the final two weeks only on writing, figures, references, checklist,
   packaging, and reproducibility.

## Paper Integration

Main text should stay within 8 pages:

1. Keep the method and mathematical analysis concise.
2. Add one compact matched-baseline table or paragraph.
3. Add destructive ablation evidence where it directly supports the mechanism.
4. Add one graph stress paragraph in limitations or analysis.
5. Put extended VAE, graph, and baseline tables into supplement/checklist notes
   unless they are decisive.

Recommended section roles:

| Section | Role |
|---|---|
| Introduction | Motivate one-step conditional generation and decoder-coupled drift. |
| Method | Define two-stage SELFIES VAE, generator, coupled drift, diversity. |
| Mathematical Analysis | Prove decoder pullback gradient, metric interpretation, bounded drift, mean-shift view. |
| Experiments | Main QED, multi-property, matched baselines, destructive ablations. |
| Analysis | Explain mechanism and representation stress. |
| Limitations | State SELFIES validity, graph bottleneck, and non-SOTA boundary. |

## Completion Gate

The extension phase is complete because all items now pass:

1. `python scripts/audit_publication_completion.py --run-tests` still passes.
2. `python scripts/audit_extension_completion.py --strict` passes.
3. `python scripts/audit_drifting_faithfulness.py` passes or explicitly records
   which faithful-Drifting runs were deferred and why.
4. Destructive ablations provide at least three interpretable negative controls.
5. VAE sensitivity has at least one credible alternative setting or a clearly
   documented failure mode.
6. Graph stress is either complete or intentionally scoped as a diagnostic
   limitation with artifacts.
7. Matched baselines include at least one trained nontrivial baseline, or the
   paper explicitly marks this as a limitation.
8. Quantitative figures and tables are regenerated from scripts and artifacts.
9. The final main PDF remains at or under 8 pages.
10. `python -m unittest discover -s tests` passes after all script changes.

## Go/No-Go Rules

Proceed with the current title and main claim if:

- DriftingMol remains clearly above trained baselines on the main control/diversity
  tradeoff, or
- trained baselines are close but DriftingMol has a clear one-step efficiency
  and mechanism-ablation advantage.

Narrow the claim if:

- VAE sensitivity shows the effect depends strongly on one VAE architecture.
- graph stress remains incomplete and cannot be presented as a clean diagnostic.
- trained baselines match DriftingMol on control, novelty, and diversity.

Do not delay submission for:

- full external SOTA reruns,
- ZINC-scale graph generation,
- large model scaling,
- purely cosmetic experiments.
