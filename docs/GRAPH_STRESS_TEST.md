# Graph Representation Stress Test

Updated: 2026-05-15 UTC

This note summarizes archived graph VAE-line experiments. It is a diagnostic
artifact, not a claim that graph generation should replace the SELFIES main
track.

## Main-Paper Anchor

- SELFIES DriftingMol (F): validity 100.0%, uniqueness 94.7%, novelty 100.0%, QED $\rho$ 0.493, MAE 0.200

## Archived Generation Quality

| Family | Run | Mode | Validity | Uniqueness | Novelty | Note |
|---|---|---|---:|---:|---:|---|
| Graph VAE prior | `vae_v2_kl01` | generation | 87.4% | 98.1% | 100.0% | Valid and novel, but the prior is still sensitive to VAE training. |
| Graph VAE prior | `vae_v3_valence` | generation | 100.0% | 74.6% | 76.4% | Validity reaches 100%, but uniqueness and novelty drop sharply. |
| Graph drift | `drifting_v2kl01_fix2` | generation_uncond | 93.7% | 96.2% | 100.0% | Best unconditional graph-drift quality in the archive. |
| Decoder drift | `e34_decoder_drift_v3` | generation_uncond | 100.0% | 47.9% | 71.5% | Perfect validity, but diversity and novelty fall off. |
| CFG graph drift | `e36_dec_drift_cfg` | generation_uncond | 100.0% | 26.9% | 77.7% | Strongest graph control in the archive, yet rho stays low and diversity collapses. |
| Phi-space drift | `E30_phi_space_drift` | generation_uncond | 92.3% | 98.7% | 100.0% | A near-null control diagnostic: validity is fine, control is almost absent. |
| LogP queue | `e40_logp_bins_queue` | generation_uncond | 100.0% | 24.3% | 72.3% | Perfect validity, but control remains weak compared with the SELFIES anchor. |
| Fresh graph QED | `e36_dec_drift_cfg_fresh` | generation_uncond | 100.0% | 13.1% | 75.4% | Fresh graph-route QED control under the same V/U/N and Spearman reporting contract. |
| Fresh graph LogP | `e40_logp_bins_queue_fresh` | generation_uncond | 100.0% | 31.9% | 75.5% | Fresh graph-route LogP control for property-transfer comparison. |
| Fresh graph ablation | `e36_no_drift_fresh` | generation_uncond | 100.0% | 12.4% | 74.3% | No-drift graph control ablation to isolate the effect of latent drifting. |

## Control Summary

| Family | Run | Target | Best $\rho$ | $\rho$ range | Gap range | Note |
|---|---|---|---:|---:|---:|---|
| CFG graph drift | `e36_dec_drift_cfg` | QED | 0.159 | 0.130 to 0.159 | 0.030 to 0.033 | Strongest graph control in the archive, yet rho stays low and diversity collapses. |
| Phi-space drift | `E30_phi_space_drift` | QED | 0.014 | -0.067 to 0.014 | -0.003 to 0.003 | A near-null control diagnostic: validity is fine, control is almost absent. |
| LogP queue | `e40_logp_bins_queue` | LogP | 0.145 | 0.076 to 0.145 | 0.345 to 0.444 | Perfect validity, but control remains weak compared with the SELFIES anchor. |
| Fresh graph QED | `e36_dec_drift_cfg_fresh` | QED | 0.019 | -0.003 to 0.019 | -0.001 to 0.005 | Fresh graph-route QED control under the same V/U/N and Spearman reporting contract. |
| Fresh graph LogP | `e40_logp_bins_queue_fresh` | LogP | 0.327 | 0.272 to 0.327 | 0.740 to 0.839 | Fresh graph-route LogP control for property-transfer comparison. |
| Fresh graph ablation | `e36_no_drift_fresh` | QED | 0.046 | -0.018 to 0.046 | -0.002 to 0.001 | No-drift graph control ablation to isolate the effect of latent drifting. |
| SELFIES anchor | `DriftingMol (F)` | QED | 0.493 | 0.493 to 0.493 | n/a | Main-paper anchor for comparison against graph stress results. |

## Raw-vs-Repaired Decoding

This diagnostic separates sanitization repair from actual control. Repair always
restores validity to 100%, but uniqueness gains are small and can turn slightly
negative at higher temperatures.

| Family | Temperature | Raw validity | Repaired validity | Raw uniqueness | Repaired uniqueness | $\Delta$validity | $\Delta$uniqueness |
|---|---:|---:|---:|---:|---:|---:|---:|
| e36_dec_drift_cfg_fresh | 0.0 | 75.8% | 100.0% | 11.4% | 12.8% | 0.242 | 0.014 |
| e36_dec_drift_cfg_fresh | 0.5 | 36.3% | 100.0% | 86.3% | 81.9% | 0.637 | -0.044 |
| e36_dec_drift_cfg_fresh | 1.0 | 18.4% | 100.0% | 99.0% | 96.7% | 0.816 | -0.023 |
| e40_logp_bins_queue_fresh | 0.0 | 78.1% | 100.0% | 30.0% | 33.6% | 0.219 | 0.036 |
| e40_logp_bins_queue_fresh | 0.5 | 40.2% | 100.0% | 94.8% | 92.5% | 0.598 | -0.022 |
| e40_logp_bins_queue_fresh | 1.0 | 22.8% | 100.0% | 99.4% | 98.2% | 0.772 | -0.012 |

## Reading

1. Graph validity is not the main issue. The harder problem is stable diversity
   and meaningful target control.
2. Stronger graph control variants still sit well below the SELFIES anchor in $\rho$.
3. For the AAAI draft, graph results are best used as a limitation / diagnostic
   appendix, not as a replacement for the current SELFIES story.

## Recommendation

- Keep SELFIES as the main method in the submission package.
- If graph work is continued, run one clean QM9 graph-control pass and one
  destructive graph ablation; do not expand into a full second method line.
