# Reviewer Feedback Synthesis

Objective: collect three independent reviewer-style assessments, evaluate the
main risks, and map the accepted feedback to concrete paper changes.

## Reviewer Scores

| Reviewer | Lens | Score | Confidence | Recommendation |
|---|---|---:|---:|---|
| Reviewer 1 | ML rigor and experimental protocol | 5.0/10 | 4/5 | Weak reject |
| Reviewer 2 | Molecular generation and SELFIES validity | 5.0/10 | 4/5 | Weak reject |
| Reviewer 3 | AAAI writing, figures, and narrative | 5.5/10 | 4/5 | Weak reject |

## Consensus Risks

| Risk | Assessment | Action |
|---|---|---|
| Overclaiming versus external baselines | The evidence is strongest as a mechanism ablation, not as a molecular-generation SOTA claim. | Repositioned the paper as a mechanism study, moved LIMO/FREED/MOOD comparison to limitations/future work, and added same-backbone generative baselines as the protocol-matched comparison. |
| SELFIES validity claim | Validity is inherited from representation and is not the method's main contribution. | Reworded abstract, metrics, and limitations to treat validity as a representation guarantee; clarified RDKit failures count against validity. |
| Alpha and seed selection | Best-alpha single-run ordering can overstate fine rank differences. | Added evaluation protocol text, emphasized pre-specified gates and tier-level conclusions, and added the three-seed aggregate/CI figure. |
| Missing calibration and diversity context | Rho alone is insufficient to judge target tracking and library quality. | Added slope to the QED table/figure labels and highlighted novelty/scaffold diversity in text. |
| Four-property table inconsistency | G4 and uniqueness needed to be visible in the fair multi-property table. | Updated table export to include G4 and the lowest property-wise uniqueness column; regenerated Table 3 and Figure 3. |
| P1/P2 faithfulness ambiguity | Paper-style audit settings differ from the Full setting and should not be mistaken for the main method. | Renamed the section to implementation-sensitivity checks and explained the prop-only, 64-reference P1/P2 protocol. |
| Drifting reproduction fidelity | A reviewer may argue that DriftingMol is not a faithful reproduction of Drifting Models before the decoder-coupled extension. | Completed Algorithm-2 code audit, 4/4 strict core runs, and the 10/10 reviewer-faithful manifest; the strict faithfulness audit now reports PASS. |
| Feature extractor dependence | The original Drifting paper emphasizes pretrained feature quality; molecular results must show what happens with latent-MAE phi, random phi, and z-space. | Completed strict plain-phi, property-aware phi, random-phi, and z-space controls; results imply modest faithful control and weak feature-superiority evidence. |
| Positive/negative sample estimation | The original paper reports Table-2 sensitivity to Npos/Nneg; reviewers may ask whether molecule results obey the same estimator logic. | Completed the full positive/negative allocation sweep; diversity recovers with larger sample sets, but target tracking remains weak, supporting a representation-limited interpretation. |
| Figure quality and density | Result figures needed cleaner academic style with less cramped annotation. | Rebuilt result figures with horizontal bars, direct labels, compact heatmap, and scatter-only z-diversity sweep. |
| Graph route launchability | A reviewer may ask whether the archived graph route can be relaunched as a limitation check. | Completed a fresh graph stress package with graph VAE checkpoint, graph latent cache, graph Latent-MAE checkpoint, fresh graph QED/LogP drifting, no-drift graph ablation, raw-vs-repaired validity, and graph-vs-SELFIES comparison. |
| Internal project language | Terms such as publication queues and repository audits are inappropriate in the paper. | Removed internal status language from the manuscript and replaced it with normal scientific limitations. |

## Implemented Files

- `docs/PAPER_AAAI.tex`: claim calibration, protocol transparency, limitations,
  sensitivity-check wording, VAE-prior sanity check, and updated
  tables/captions.
- `scripts/export_latex_tables.py`: synchronized QED slope and multi-property G4
  / lowest-uniqueness columns with generated LaTeX tables.
- `scripts/plot_result_figures.py`: regenerated cleaner result figures for QED,
  multi-property conditioning, seed stability, and z-diversity.
- `results/tables/tab_qed_main.tex`, `results/tables/tab_multi4_v2.tex`:
  refreshed table artifacts.
- `docs/figures/fig2_qed_ablation.*`, `docs/figures/fig3_multi4_v2.*`,
  `docs/figures/fig4_qed_seed_ci.*`, `docs/figures/fig5_zdiv_pareto.*`:
  refreshed figure artifacts.
- `docs/DRIFTING_FAITHFULNESS_PLAN.md`: reviewer-facing plan for faithful
  Drifting reproduction and supplemental experiments.
- `configs/reviewer_faithful/manifest.json`: strict Drifting-faithfulness
  core and allocation configs.
- `scripts/generate_faithful_drifting_configs.py`,
  `scripts/collect_faithful_drifting_results.py`,
  `scripts/audit_drifting_faithfulness.py`,
  `scripts/defer_faithful_core_after_destructive.py`: generation,
  collection, audit, and deferred launch automation.
- `results/drifting_faithfulness_audit.md` and
  `results/faithful_drifting_status.json`: current faithfulness status.
- `scripts/train_latent_generative_baseline.py`,
  `configs/publication_ext/generative_baselines_manifest.json`,
  `results/generative_baselines_qed.json`, and
  `results/tables/tab_generative_baselines_qed.tex`: same-backbone
  CVAE/WGAN-GP/DDPM/Flow-Matching generator-family baselines.

## Iteration Log

| Round | Main changes | Estimated score |
|---|---|---:|
| Initial reviewer round | Three independent reviews identified overclaiming, missing external baselines, weak metric reporting, table inconsistency, and figure-density issues. | 5.0--5.5 |
| Revision 1 | Repositioned as a mechanism ablation; softened validity and SOTA claims; fixed the multi-property uniqueness-table mismatch; regenerated result figures; added protocol and sensitivity-check text. | 6.5--7.0 |
| Revision 2 | Added explicit QED diversity profile with V/N/IntDiv/scaffold diversity, clarified protocol-mismatch with optimization methods, and added the VAE-prior sanity check separating SELFIES validity from conditional control. | 7.5--8.0 |
| Independent review round | Three new independent reviewers scored the revised draft 5.0, 5.8, and 6.0. Consensus: the paper still needed protocol-matched references, fixed-alpha/per-target transparency, and AAAI checklist readiness. | 5.0--6.0 |
| Revision 3 | Added matched QED retrieval and frozen-VAE latent-jitter references, documented their low novelty despite high target matching, added the fixed split, and prepared the AAAI reproducibility checklist as a separate file. | pending rereview |
| Revision 4 execution | Completed the Algorithm-2 audit, strict core faithful-Drifting runs, full allocation sweep, VAE architecture sensitivity, downstream VAE-drift checks, destructive ablations, and the three-seed fixed linear property-guidance baseline. The evidence supports a tighter claim: faithful direct Drifting is measurable but weak in molecule space, while decoder-coupled DriftingMol carries the useful control signal. | pending generalization/reviewer-extra |
| Revision 5 evidence pack | Completed reviewer-extra robustness, next-wave fixed-guidance controls, full graph stress, and the same-backbone generator-family baselines. The paper now has a stronger reviewer artifact package while keeping the main claim framed as mechanism evidence rather than SOTA molecular generation. | 7.5--8.0 |

## Residual Limitations

The revised submission still should be presented as a compact mechanism paper.
It now includes same-backbone generative baselines and full graph stress
evidence, but it does not claim broad external SOTA comparison against
iterative molecular optimization systems. The main QED tier remains three-seed
rather than a larger seed sweep.
