from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _load_collect_results():
    path = Path(__file__).resolve().parents[1] / "scripts" / "collect_results.py"
    spec = importlib.util.spec_from_file_location("collect_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CollectResultsTest(unittest.TestCase):
    def test_qed_best_alpha_ignores_collapsed_uniqueness(self):
        collect = _load_collect_results()
        metrics = {
            "alpha=1.0": {
                "conditional_qed": {
                    "validity": 1.0,
                    "uniqueness": 0.01,
                    "novelty": 1.0,
                    "spearman_rho": 0.9,
                }
            },
            "alpha=3.0": {
                "conditional_qed": {
                    "validity": 1.0,
                    "uniqueness": 0.95,
                    "novelty": 1.0,
                    "spearman_rho": 0.5,
                }
            },
        }
        alpha, summary, gate = collect.summarize_qed(metrics, 0.95, 0.10)
        self.assertEqual(alpha, "alpha=3.0")
        self.assertEqual(summary["spearman_rho"], 0.5)
        self.assertEqual(gate, "pass")

    def test_multi4_summary_uses_average_rho(self):
        collect = _load_collect_results()
        entry = {
            "conditional_qed": {"validity": 1.0, "uniqueness": 0.9, "novelty": 1.0, "spearman_rho": 0.2, "mae": 0.1, "slope": 0.2},
            "conditional_sa_score": {"validity": 1.0, "uniqueness": 0.9, "novelty": 1.0, "spearman_rho": 0.4, "mae": 1.0, "slope": 0.4},
            "conditional_logp": {"validity": 1.0, "uniqueness": 0.9, "novelty": 1.0, "spearman_rho": 0.6, "mae": 1.0, "slope": 0.6},
            "conditional_molwt": {"validity": 1.0, "uniqueness": 0.9, "novelty": 1.0, "spearman_rho": 0.8, "mae": 50.0, "slope": 0.8},
        }
        alpha, summary, gate = collect.summarize_multi4({"alpha=5.0": entry}, 0.95, 0.10)
        self.assertEqual(alpha, "alpha=5.0")
        self.assertEqual(summary["avg_spearman_rho"], 0.5)
        self.assertEqual(summary["min_uniqueness"], 0.9)
        self.assertEqual(gate, "pass")

    def test_qed_3seed_aggregate_deduplicates_seed42(self):
        collect = _load_collect_results()
        rows = [
            collect.ExperimentRow(
                status="complete_pass",
                root="final",
                experiment="exp_F_qed",
                variant="F",
                condition="qed",
                output_dir="outputs/final/exp_F_qed",
                seed=42,
                metrics={"spearman_rho": 0.49, "uniqueness": 0.94, "mae": 0.20, "slope": 0.79},
            ),
            collect.ExperimentRow(
                status="complete_pass",
                root="seeds",
                experiment="pub_F_qed_s42",
                variant="F",
                condition="qed",
                output_dir="outputs/publication/seeds/pub_F_qed_s42",
                seed=42,
                manifest_group="qed_3seed",
                metrics={"spearman_rho": 0.51, "uniqueness": 0.95, "mae": 0.19, "slope": 0.80},
            ),
            collect.ExperimentRow(
                status="complete_pass",
                root="seeds",
                experiment="pub_F_qed_s43",
                variant="F",
                condition="qed",
                output_dir="outputs/publication/seeds/pub_F_qed_s43",
                seed=43,
                manifest_group="qed_3seed",
                metrics={"spearman_rho": 0.53, "uniqueness": 0.96, "mae": 0.18, "slope": 0.82},
            ),
            collect.ExperimentRow(
                status="complete_pass",
                root="zdiv",
                experiment="pub_G4_qed_zdiv0p0_s42",
                variant="G4",
                condition="qed",
                output_dir="outputs/publication/zdiv/pub_G4_qed_zdiv0p0_s42",
                seed=42,
                manifest_group="zdiv_pareto",
                metrics={"spearman_rho": 0.99, "uniqueness": 0.01, "mae": 0.1, "slope": 1.0},
            ),
        ]
        summary = collect.aggregate_qed_3seed(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["variant"], "F")
        self.assertEqual(summary[0]["seeds"], "42,43")
        self.assertEqual(summary[0]["n"], 2)
        self.assertAlmostEqual(summary[0]["rho_mean"], 0.52)

    def test_manifest_row_extracts_seed_from_name_without_yaml(self):
        collect = _load_collect_results()
        row = collect.manifest_row({
            "group": "qed_3seed",
            "name": "pub_A6_qed_s44",
            "config": "missing.yaml",
            "output_dir": "outputs/publication/seeds/pub_A6_qed_s44",
        })

        self.assertEqual(row.status, "pending")
        self.assertEqual(row.manifest_group, "qed_3seed")
        self.assertEqual(row.variant, "A6")
        self.assertEqual(row.condition, "qed")
        self.assertEqual(row.seed, 44)

    def test_extension_manifest_row_parses_ext_name(self):
        collect = _load_collect_results()
        row = collect.manifest_row({
            "group": "destructive_drift",
            "name": "ext_D_ATTR_qed_s42",
            "config": "missing.yaml",
            "output_dir": "outputs/publication_ext/destructive/ext_D_ATTR_qed_s42",
        })

        self.assertEqual(row.status, "pending")
        self.assertEqual(row.manifest_group, "destructive_drift")
        self.assertEqual(row.variant, "D_ATTR")
        self.assertEqual(row.condition, "qed")
        self.assertEqual(row.seed, 42)

    def test_extension_manifest_row_parses_vae_name(self):
        collect = _load_collect_results()
        row = collect.manifest_row({
            "group": "vae_sensitivity",
            "name": "ext_V_BETA_LOW_vae_s42",
            "config": "missing.yaml",
            "output_dir": "outputs/publication_ext/vae_sensitivity/ext_V_BETA_LOW_vae_s42",
        })

        self.assertEqual(row.variant, "V_BETA_LOW")
        self.assertEqual(row.condition, "vae")
        self.assertEqual(row.seed, 42)

    def test_canonical_seed42_detection(self):
        collect = _load_collect_results()
        self.assertTrue(collect.covered_by_canonical_seed42({
            "group": "qed_3seed",
            "output_dir": "outputs/publication/seeds/pub_F_qed_s42",
        }))
        self.assertFalse(collect.covered_by_canonical_seed42({
            "group": "qed_3seed",
            "output_dir": "outputs/publication/seeds/pub_F_qed_s43",
        }))

    def test_status_payload_lists_pending_entries(self):
        collect = _load_collect_results()
        rows = [
            collect.ExperimentRow(
                status="complete_pass",
                root="final",
                experiment="exp_F_qed",
                variant="F",
                condition="qed",
                output_dir="outputs/final/exp_F_qed",
                seed=42,
                metrics={"spearman_rho": 0.49, "uniqueness": 0.94, "mae": 0.20, "slope": 0.79},
            ),
            collect.ExperimentRow(
                status="pending",
                root="seeds",
                experiment="pub_F_qed_s43",
                variant="F",
                condition="qed",
                output_dir="outputs/publication/seeds/pub_F_qed_s43",
                seed=43,
                manifest_group="qed_3seed",
            ),
        ]

        status = collect.build_status(rows)

        self.assertEqual(status["num_experiments"], 2)
        self.assertEqual(status["complete"], 1)
        self.assertEqual(status["pending_or_incomplete"], 1)
        self.assertEqual(status["status_counts"], {"complete_pass": 1, "pending": 1})
        self.assertEqual(len(status["pending_or_incomplete_entries"]), 1)
        self.assertEqual(status["pending_or_incomplete_entries"][0]["experiment"], "pub_F_qed_s43")

    def test_running_status_marks_pending_row_active_before_checkpoint(self):
        collect = _load_collect_results()
        row = collect.ExperimentRow(
            status="pending",
            root="zdiv",
            experiment="pub_G4_qed_zdiv0p0_s42",
            variant="G4",
            condition="qed",
            output_dir="outputs/publication/zdiv/pub_G4_qed_zdiv0p0_s42",
            seed=42,
            manifest_group="zdiv_pareto",
        )

        with tempfile.TemporaryDirectory() as tmp:
            status_root = Path(tmp)
            (status_root / "runner_status_followup_gpu0.json").write_text(json.dumps({
                "state": "running",
                "entry": {
                    "output_dir": "outputs/publication/zdiv/pub_G4_qed_zdiv0p0_s42",
                },
            }))

            collect.apply_running_status([row], status_root)

        self.assertEqual(row.status, "running_or_incomplete")


if __name__ == "__main__":
    unittest.main()
