import json
import tempfile
import unittest
from pathlib import Path

import scripts.collect_trained_baselines as collector


class CollectTrainedBaselinesTest(unittest.TestCase):
    def _payload(self, seed: int, status: str):
        return {
            "status": status,
            "group": "trained_baseline",
            "experiment": f"ext_B_LINEAR_PROP_QED_s{seed}",
            "display": "Linear property guidance",
            "purpose": "",
            "seed": seed,
            "alpha": "alpha=3.0",
            "spearman_rho": 0.1 + seed / 1000.0,
            "validity": 1.0,
            "uniqueness": 0.95,
            "novelty": 1.0,
            "mae": 0.2,
            "slope": 0.1,
            "warning": "",
            "output_dir": "",
            "config": "",
            "command": "",
        }

    def test_status_tracks_three_seed_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "configs/publication_ext/baseline_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            out = root / "results/status.json"
            payloads = [
                self._payload(42, "complete_pass"),
                self._payload(43, "running_or_incomplete"),
                self._payload(44, "pending"),
            ]

            old_root = collector.ROOT
            try:
                collector.ROOT = root
                collector.write_status(out, manifest, payloads)
            finally:
                collector.ROOT = old_root

            status = json.loads(out.read_text())

        self.assertEqual(status["complete"], 1)
        self.assertFalse(status["three_seed_complete"])
        self.assertEqual(status["aggregate"]["n"], 1)
        self.assertEqual(len(status["pending_or_incomplete_entries"]), 2)

    def test_parallel_running_status_marks_named_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_root = root / "outputs/publication_ext"
            status_root.mkdir(parents=True)
            (status_root / "parallel_runner_status_baseline_s43.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "running": [{"name": "ext_B_LINEAR_PROP_QED_s43"}],
                    }
                )
            )
            rows = [
                collector.collect.ExperimentRow(
                    status="pending",
                    root="baselines",
                    experiment="ext_B_LINEAR_PROP_QED_s43",
                    variant="ext_B_LINEAR_PROP_QED_s43",
                    condition="unknown",
                    output_dir="outputs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s43",
                )
            ]

            collector.apply_parallel_running(rows, status_root)

        self.assertEqual(rows[0].status, "running_or_incomplete")

    def test_tex_includes_pending_and_summary_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "table.tex"
            payloads = [
                self._payload(42, "complete_pass"),
                self._payload(43, "pending"),
            ]
            payloads[1]["spearman_rho"] = None
            payloads[1]["uniqueness"] = None
            payloads[1]["mae"] = None
            payloads[1]["slope"] = None

            collector.write_tex(out, payloads)
            text = out.read_text()

        self.assertIn("complete\\_pass", text)
        self.assertIn("pending", text)
        self.assertIn("mean (1 seeds)", text)


if __name__ == "__main__":
    unittest.main()
