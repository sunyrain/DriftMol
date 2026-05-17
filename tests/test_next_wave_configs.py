import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class NextWaveConfigTest(unittest.TestCase):
    def test_manifest_entries_and_configs_match_reviewer_purpose(self):
        manifest_path = ROOT / "configs/publication_ext/next_wave_manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text())
        entries = manifest["entries"]

        self.assertEqual(len(entries), 4)
        self.assertEqual(
            {entry["group"] for entry in entries},
            {"property_guidance_baseline", "conditioning_seed_stability"},
        )
        for entry in entries:
            self.assertTrue((ROOT / entry["config"]).exists(), entry["config"])
            self.assertFalse(manifest["resource_policy"]["launch_now"])

    def test_logp_and_multi4_linear_baselines_disable_drift(self):
        logp_cfg = yaml.safe_load(
            (ROOT / "configs/publication_ext/next_wave/ext_NW_LINEAR_PROP_LOGP_s42.yaml").read_text()
        )
        multi_cfg = yaml.safe_load(
            (ROOT / "configs/publication_ext/next_wave/ext_NW_LINEAR_PROP_MULTI4_s42.yaml").read_text()
        )

        self.assertEqual(logp_cfg["data"]["prop_indices"], [2])
        self.assertEqual(logp_cfg["generator"]["cond_dim"], 1)
        self.assertEqual(logp_cfg["loss"]["lambda_drift"], 0.0)
        self.assertEqual(logp_cfg["loss"]["lambda_prop"], 1.0)
        self.assertEqual(logp_cfg["prop_head"]["type"], "linear")

        self.assertEqual(multi_cfg["data"]["prop_indices"], [0, 1, 2, 3])
        self.assertEqual(multi_cfg["generator"]["cond_dim"], 4)
        self.assertEqual(multi_cfg["loss"]["lambda_drift"], 0.0)
        self.assertEqual(multi_cfg["loss"]["lambda_prop"], 1.0)

    def test_continuous_qed_second_seed_disables_binning(self):
        cfg = yaml.safe_load(
            (ROOT / "configs/publication_ext/next_wave/ext_NW_QED_CONTINUOUS_s43.yaml").read_text()
        )

        self.assertEqual(cfg["experiment"]["seed"], 43)
        self.assertFalse(cfg["cond_binning"]["enabled"])
        self.assertEqual(cfg["cfg"]["positive_mode"], "prop")


if __name__ == "__main__":
    unittest.main()
