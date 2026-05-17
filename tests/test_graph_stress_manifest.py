import json
import unittest
from pathlib import Path


class GraphStressManifestTest(unittest.TestCase):
    def test_manifest_has_recovery_and_fresh_rows(self):
        path = Path("configs/publication_ext/graph_stress_manifest.json")
        payload = json.loads(path.read_text())

        self.assertTrue(payload["resource_policy"]["launch_now"])
        self.assertGreaterEqual(len(payload["preconditions"]), 4)
        self.assertEqual(len(payload["archived_diagnostics"]), 2)
        self.assertEqual(len(payload["entries"]), 9)

        names = {entry["name"] for entry in payload["entries"]}
        self.assertIn("graph_rebuild_qm9_graph_cache", names)
        self.assertIn("graph_recover_vae_v3_valence", names)
        self.assertIn("graph_rebuild_latent_cache_v3", names)
        self.assertIn("graph_recover_latent_mae_v3", names)
        self.assertIn("graph_fresh_qed_e36", names)
        self.assertIn("graph_fresh_logp_e40", names)
        self.assertIn("graph_raw_vs_repaired_decode", names)
        self.assertIn("graph_selfies_fair_comparison", names)


if __name__ == "__main__":
    unittest.main()
