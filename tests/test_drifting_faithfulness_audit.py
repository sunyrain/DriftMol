import json
import tempfile
import unittest
from pathlib import Path

import yaml

import scripts.audit_drifting_faithfulness as audit


class DriftingFaithfulnessAuditTest(unittest.TestCase):
    def _base_cfg(self):
        return {
            "training": {"epochs": 100},
            "loss": {
                "lambda_drift": 1.0,
                "lambda_zdrift": 0.0,
                "lambda_decoupled_drift": 0.0,
                "lambda_dec_drift": 0.0,
                "lambda_zdiv": 0.0,
                "lambda_phidiv": 0.0,
                "temperatures": [0.02, 0.05, 0.2],
                "drift_normalize": True,
                "drift_normalize_dist": True,
                "drift_norm_mode": "xy",
                "drift_attraction_scale": 1.0,
                "drift_repulsion_scale": 1.0,
            },
            "cfg": {
                "n_groups": 64,
                "n_gen": 64,
                "n_pos": 64,
                "positive_mode": "prop",
                "alpha_power": 3,
                "alpha_min": 1.0,
                "alpha_max": 4.0,
            },
            "cond_binning": {"enabled": True, "method": "quantile"},
            "feature_space": {"mode": "phi"},
            "phi": {"checkpoint": "outputs/foundation/zinc_phi_plain/best_latent_mae.pt"},
        }

    def _write_cfg(self, root: Path, name: str, group: str, cfg: dict) -> dict:
        path = root / "configs/reviewer_faithful" / group / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return {"name": name, "group": group, "config": str(path.relative_to(root))}

    def _prepare_manifest(self, root: Path, *, zdiv: float = 0.0):
        entries = []
        specs = [
            (
                "rf_FD_STRICT_PLAIN_PHI_QED_s42",
                "faithful_core",
                {"feature_space": {"mode": "phi"}, "phi": {"checkpoint": "outputs/foundation/zinc_phi_plain/best_latent_mae.pt"}},
            ),
            (
                "rf_FD_STRICT_PROP_PHI_QED_s42",
                "faithful_core",
                {"feature_space": {"mode": "phi"}, "phi": {"checkpoint": "outputs/foundation/zinc_phi_prop/best_latent_mae.pt"}},
            ),
            (
                "rf_FD_STRICT_RANDOM_PHI_QED_s42",
                "faithful_core",
                {"feature_space": {"mode": "random"}, "phi": {"checkpoint": ""}},
            ),
            (
                "rf_FD_STRICT_ZSPACE_QED_s42",
                "faithful_core",
                {"loss": {"lambda_drift": 0.0, "lambda_zdrift": 1.0}, "phi": {"checkpoint": ""}},
            ),
        ]
        for name, group, updates in specs:
            cfg = self._base_cfg()
            cfg["loss"]["lambda_zdiv"] = zdiv
            for section, values in updates.items():
                cfg.setdefault(section, {}).update(values)
            entries.append(self._write_cfg(root, name, group, cfg))

        for idx in range(6):
            cfg = self._base_cfg()
            cfg["loss"]["lambda_zdiv"] = zdiv
            entries.append(self._write_cfg(root, f"alloc_{idx}", "faithful_allocation", cfg))

        manifest = root / "configs/reviewer_faithful/manifest.json"
        manifest.write_text(json.dumps({"entries": entries}))

    def _with_root(self, root: Path):
        old_root, old_manifest = audit.ROOT, audit.MANIFEST
        audit.ROOT = root
        audit.MANIFEST = root / "configs/reviewer_faithful/manifest.json"
        return old_root, old_manifest

    def _restore_root(self, saved):
        audit.ROOT, audit.MANIFEST = saved

    def test_strict_protocol_config_status_passes_for_valid_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_manifest(root)
            saved = self._with_root(root)
            try:
                status = audit.strict_protocol_config_status()
            finally:
                self._restore_root(saved)

        self.assertEqual(status["status"], "PASS")
        self.assertEqual(status["checked"], 10)
        self.assertEqual(status["failures"], [])

    def test_strict_protocol_config_status_rejects_zdiv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_manifest(root, zdiv=0.5)
            saved = self._with_root(root)
            try:
                status = audit.strict_protocol_config_status()
            finally:
                self._restore_root(saved)

        self.assertEqual(status["status"], "OPEN")
        self.assertTrue(any("z-diversity enabled" in failure for failure in status["failures"]))


if __name__ == "__main__":
    unittest.main()
