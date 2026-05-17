import json
import tempfile
import unittest
from pathlib import Path

import scripts.collect_vae_drift_results as collector


class CollectVaeDriftResultsTest(unittest.TestCase):
    def _with_root(self, root: Path):
        saved = (
            collector.ROOT,
            collector.MANIFEST,
            collector.OUT_CSV,
            collector.OUT_STATUS,
            collector.OUT_TEX,
        )
        collector.ROOT = root
        collector.MANIFEST = root / "configs/publication_ext/vae_drift_manifest.json"
        collector.OUT_CSV = root / "results/vae_drift_downstream.csv"
        collector.OUT_STATUS = root / "results/vae_drift_downstream_status.json"
        collector.OUT_TEX = root / "results/tables/tab_vae_drift_downstream.tex"
        return saved

    def _restore(self, saved):
        (
            collector.ROOT,
            collector.MANIFEST,
            collector.OUT_CSV,
            collector.OUT_STATUS,
            collector.OUT_TEX,
        ) = saved

    def test_pending_row_includes_upstream_vae_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vae_metrics = root / "outputs/vae/final_metrics.json"
            vae_metrics.parent.mkdir(parents=True)
            vae_metrics.write_text(json.dumps({"exact_recon": 0.9, "prior_vun": 0.8}))
            entry = {
                "name": "ext_vae_lowbeta_drift_qed_s42",
                "display": "Low-beta VAE",
                "output_dir": "outputs/drift/lowbeta",
                "depends_on": ["outputs/vae/final_metrics.json"],
            }
            saved = self._with_root(root)
            try:
                row = collector.row_for_entry(entry, running=set())
            finally:
                self._restore(saved)

        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["vae_exact_recon"], 0.9)
        self.assertEqual(row["vae_prior_vun"], 0.8)

    def test_best_qed_prefers_quality_gated_alpha(self):
        metrics = {
            "alpha=1.0": {
                "conditional_qed": {
                    "spearman_rho": 0.2,
                    "validity": 1.0,
                    "uniqueness": 0.95,
                    "novelty": 1.0,
                    "mae": 0.2,
                }
            },
            "alpha=3.0": {
                "conditional_qed": {
                    "spearman_rho": 0.9,
                    "validity": 1.0,
                    "uniqueness": 0.5,
                    "novelty": 1.0,
                    "mae": 0.1,
                }
            },
        }

        alpha, section, gate = collector.best_qed(metrics)

        self.assertEqual(alpha, "alpha=1.0")
        self.assertEqual(section["spearman_rho"], 0.2)
        self.assertEqual(gate, "pass")

    def test_status_counts_complete_and_pending_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = self._with_root(root)
            try:
                collector.write_status(
                    [
                        {"status": "complete_pass", "experiment": "done"},
                        {"status": "pending", "experiment": "todo"},
                    ]
                )
                status = json.loads(collector.OUT_STATUS.read_text())
            finally:
                self._restore(saved)

        self.assertEqual(status["complete"], 1)
        self.assertEqual(status["pending_or_incomplete"], 1)
        self.assertTrue(status["minimum_completed_runs_reached"])


if __name__ == "__main__":
    unittest.main()
