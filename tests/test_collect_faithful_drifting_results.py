import json
import tempfile
import unittest
from pathlib import Path

import scripts.collect_faithful_drifting_results as collector


class CollectFaithfulDriftingResultsTest(unittest.TestCase):
    def _payload(self, group: str, name: str, status: str):
        return {
            "status": status,
            "group": group,
            "experiment": name,
            "display": name,
            "purpose": "",
            "seed": 42,
            "alpha": "alpha=3.0",
            "spearman_rho": 0.5,
            "validity": 1.0,
            "uniqueness": 0.95,
            "novelty": 1.0,
            "mae": 0.1,
            "slope": 0.8,
            "warning": "",
            "config": "",
            "output_dir": "",
            "command": "",
        }

    def test_write_status_marks_core_complete_independently_of_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "configs/reviewer_faithful/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            out = root / "results/status.json"
            payloads = [
                self._payload("faithful_core", f"core_{idx}", "complete_pass")
                for idx in range(4)
            ]
            payloads.append(self._payload("faithful_allocation", "alloc_0", "pending"))

            old_root = collector.ROOT
            try:
                collector.ROOT = root
                collector.write_status(out, manifest, payloads)
            finally:
                collector.ROOT = old_root

            status = json.loads(out.read_text())

        self.assertTrue(status["faithful_core_complete"])
        self.assertEqual(status["groups"]["faithful_core"]["complete"], 4)
        self.assertEqual(status["groups"]["faithful_allocation"]["pending"], 1)
        self.assertEqual(status["pending_or_incomplete"], 1)

    def test_write_tex_formats_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "table.tex"
            collector.write_tex(
                out,
                [self._payload("faithful_core", "Strict plain-$\\phi$", "complete_pass")],
                "unit",
            )
            text = out.read_text()

        self.assertIn("complete\\_pass", text)
        self.assertIn("0.500", text)
        self.assertIn("95.0", text)

    def test_write_checklist_includes_gates_and_audit_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "checklist.md"
            payloads = [
                self._payload("faithful_core", "core_done", "complete_pass"),
                self._payload("faithful_core", "core_pending", "pending"),
                self._payload("faithful_allocation", "alloc_pending", "pending"),
            ]

            collector.write_checklist(out, payloads)
            text = out.read_text()

        self.assertIn("Strict faithful core: 1/2 complete", text)
        self.assertIn("Allocation sweeps: 0/1 complete", text)
        self.assertIn("defer_faithful_core_after_destructive.py", text)
        self.assertIn("audit_drifting_faithfulness.py", text)

    def test_apply_running_names_marks_parallel_status_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_root = root / "outputs/reviewer_faithful"
            status_root.mkdir(parents=True)
            (status_root / "allocation_status.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "running": [{"name": "alloc_running"}],
                    }
                )
            )
            payloads = [
                self._payload("faithful_allocation", "alloc_running", "pending"),
                self._payload("faithful_allocation", "alloc_waiting", "pending"),
            ]

            collector.apply_running_names(payloads, status_root)

        by_name = {payload["experiment"]: payload for payload in payloads}
        self.assertEqual(by_name["alloc_running"]["status"], "running_or_incomplete")
        self.assertEqual(by_name["alloc_waiting"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
