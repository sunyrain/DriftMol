import json
import tempfile
import unittest
from pathlib import Path

import scripts.audit_extension_completion as audit


class ExtensionCompletionAuditTest(unittest.TestCase):
    def _with_root(self, root: Path):
        old_root, old_results, old_md, old_json = audit.ROOT, audit.RESULTS, audit.OUT_MD, audit.OUT_JSON
        audit.ROOT = root
        audit.RESULTS = root / "results"
        audit.OUT_MD = root / "results/extension_completion_audit.md"
        audit.OUT_JSON = root / "results/extension_completion_status.json"
        return old_root, old_results, old_md, old_json

    def _restore_root(self, saved):
        audit.ROOT, audit.RESULTS, audit.OUT_MD, audit.OUT_JSON = saved

    def test_trained_baseline_check_requires_three_complete_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "configs/publication_ext/baseline_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"entries": [{"name": f"s{seed}"} for seed in [42, 43, 44]]}))
            status = root / "results/trained_baseline_status.json"
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({
                "num_experiments": 3,
                "complete": 2,
                "three_seed_complete": False,
            }))
            table = root / "results/tables/tab_trained_baseline_qed.tex"
            table.parent.mkdir(parents=True)
            table.write_text("table\n")
            saved = self._with_root(root)
            try:
                check = audit.check_trained_baselines()
            finally:
                self._restore_root(saved)

        self.assertEqual(check.status, "OPEN")
        self.assertIn("complete=2/3", check.evidence)

    def test_trained_baseline_check_passes_when_three_seed_table_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "configs/publication_ext/baseline_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"entries": [{"name": f"s{seed}"} for seed in [42, 43, 44]]}))
            status = root / "results/trained_baseline_status.json"
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({
                "num_experiments": 3,
                "complete": 3,
                "three_seed_complete": True,
            }))
            table = root / "results/tables/tab_trained_baseline_qed.tex"
            table.parent.mkdir(parents=True)
            table.write_text("table\n")
            saved = self._with_root(root)
            try:
                check = audit.check_trained_baselines()
            finally:
                self._restore_root(saved)

        self.assertEqual(check.status, "PASS")

    def test_vae_sensitivity_check_requires_all_four_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs/publication_ext/vae_sensitivity"
            cfg_dir.mkdir(parents=True)
            for idx in range(4):
                (cfg_dir / f"vae_{idx}.yaml").write_text("ok\n")
            status = root / "results/vae_sensitivity_status.json"
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({
                "num_experiments": 4,
                "complete": 3,
                "pending_or_incomplete": 1,
            }))
            saved = self._with_root(root)
            try:
                check = audit.check_vae_sensitivity()
            finally:
                self._restore_root(saved)

        self.assertEqual(check.status, "OPEN")
        self.assertIn("complete=3/4", check.evidence)

    def test_reviewer_extra_queue_check_requires_configs_and_watchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            specs = [
                ("continuous_conditioning", "cont"),
                ("single_property_seed_extension", "logp"),
                ("single_property_seed_extension", "sa"),
                ("vae_drift_seed_extension", "vae"),
            ]
            for group, name in specs:
                cfg = root / f"configs/publication_ext/reviewer_extra/{name}.yaml"
                cfg.parent.mkdir(parents=True, exist_ok=True)
                cfg.write_text("ok\n")
                entries.append({
                    "group": group,
                    "name": name,
                    "config": str(cfg.relative_to(root)),
                })
            manifest = root / "configs/publication_ext/reviewer_extra_manifest.json"
            manifest.write_text(json.dumps({"entries": entries}))
            for rel in [
                "scripts/collect_reviewer_extra_results.py",
                "scripts/watch_reviewer_extra_postprocess.py",
                "results/reviewer_extra_status.json",
                "results/tables/tab_reviewer_extra.tex",
                "outputs/publication_ext/reviewer_extra_postprocess.pid",
            ]:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok\n")
            for gpu in range(4):
                path = root / f"outputs/publication_ext/reviewer_extra_launcher_gpu{gpu}.pid"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("123\n")

            saved = self._with_root(root)
            try:
                check = audit.check_reviewer_extra_queue()
            finally:
                self._restore_root(saved)

        self.assertEqual(check.status, "PASS")


if __name__ == "__main__":
    unittest.main()
