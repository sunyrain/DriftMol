import tempfile
import unittest
import json
from pathlib import Path

import scripts.audit_graph_archive_launchability as audit


class GraphArchiveLaunchabilityTest(unittest.TestCase):
    def _write_configs(self, archive: Path) -> None:
        cfg_dir = archive / "configs"
        cfg_dir.mkdir(parents=True)
        text = """
vae:
  checkpoint: outputs/vae_v3_valence/best.pt
phi:
  checkpoint: outputs/latent_mae_v3/best_latent_mae.pt
data:
  latent_cache_path: data/cache/qm9_latent_cache_v3.pt
"""
        (cfg_dir / "e36_dec_drift_cfg.yaml").write_text(text)
        (cfg_dir / "e40_logp_bins_queue.yaml").write_text(text)

    def test_missing_artifacts_and_namespace_keep_status_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive" / "graph_vae_line"
            self._write_configs(archive)
            (archive / "src_train").mkdir()
            (archive / "src_train" / "train_generator.py").write_text(
                "from src.utils import load_vae\n"
            )
            (root / "src").mkdir()
            (root / "src" / "utils.py").write_text("# selfies utils\n")
            graph_cache = root / "archive" / "data_qm9" / "qm9_graph_cache.pt"
            graph_cache.parent.mkdir(parents=True)
            graph_cache.write_text("cache\n")
            metrics_dir = archive / "outputs" / "e36_dec_drift_cfg"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "resolved_config.json").write_text("{}\n")
            (metrics_dir / "final_metrics.json").write_text(
                json.dumps(
                    {
                        "generation_cfg_a3.0": {
                            "validity": 1.0,
                            "uniqueness": 0.25,
                            "novelty": 0.75,
                        },
                        "prop_control_a2.0": {"spearman_rho": 0.1},
                        "prop_control_a3.0": {
                            "spearman_rho": 0.2,
                            "prop_gap": 0.03,
                            "n_valid_corr": 2000,
                        },
                    }
                )
                + "\n"
            )
            vae_final = archive / "outputs" / "vae_v3_valence" / "final_metrics.json"
            vae_final.parent.mkdir(parents=True)
            vae_final.write_text("{}\n")
            mae_log = archive / "outputs" / "latent_mae_v3_train.log"
            mae_log.write_text("Checkpoint saved to outputs/latent_mae_v3/best_latent_mae.pt\n")

            status = audit.build_status(root=root, archive=archive)

        self.assertFalse(status["complete"])
        self.assertEqual(len(status["missing_required_artifacts"]), 6)
        self.assertTrue(status["namespace"]["blockers"])
        self.assertTrue(status["recovery_candidates"]["graph_cache"]["exists"])
        self.assertTrue(status["recovery_candidates"]["vae_v3_valence_final_metrics"]["exists"])
        self.assertTrue(status["recovery_candidates"]["latent_mae_v3_train_log"]["mentions_checkpoint"])
        diagnostics = status["archived_diagnostic_runs"]
        e36 = next(row for row in diagnostics if row["name"] == "e36_dec_drift_cfg")
        self.assertTrue(e36["metrics_exists"])
        self.assertEqual(e36["best_prop_control"]["alpha"], "3.0")
        self.assertAlmostEqual(e36["best_prop_control"]["spearman_rho"], 0.2)
        self.assertAlmostEqual(e36["generation_at_best_alpha"]["uniqueness"], 0.25)

    def test_complete_when_artifacts_and_archive_namespace_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive" / "graph_vae_line"
            self._write_configs(archive)
            for rel in [
                "outputs/vae_v3_valence/best.pt",
                "outputs/latent_mae_v3/best_latent_mae.pt",
                "data/cache/qm9_latent_cache_v3.pt",
            ]:
                path = archive / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok\n")
            (archive / "src_train").mkdir()
            (archive / "src_train" / "train_generator.py").write_text(
                "from src.utils import load_vae\n"
            )
            (archive / "src").mkdir()
            (archive / "src" / "utils.py").write_text("def load_vae(): pass\n")

            status = audit.build_status(root=root, archive=archive)

        self.assertTrue(status["complete"])
        self.assertEqual(status["missing_required_artifacts"], [])
        self.assertEqual(status["namespace"]["blockers"], [])


if __name__ == "__main__":
    unittest.main()
