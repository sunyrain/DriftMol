import json
import os
import tempfile
import unittest
from pathlib import Path

import scripts.monitor_extension as monitor


class MonitorExtensionTest(unittest.TestCase):
    def test_faithful_status_reports_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "faithful.json"
            path.write_text(json.dumps({
                "num_experiments": 10,
                "complete": 2,
                "pending_or_incomplete": 8,
                "faithful_core_complete": False,
                "groups": {"faithful_core": {"complete": 1, "total": 4}},
            }))
            status = monitor.faithful_status(path)

        self.assertEqual(status["complete"], 2)
        self.assertFalse(status["faithful_core_complete"])
        self.assertEqual(status["groups"]["faithful_core"]["total"], 4)

    def test_result_status_handles_trained_baseline_status_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trained_baseline_status.json"
            path.write_text(json.dumps({
                "num_experiments": 3,
                "complete": 1,
                "pending_or_incomplete": 2,
                "status_counts": {"complete_pass": 1, "pending": 2},
            }))
            status = monitor.result_status(path)

        self.assertEqual(status["complete"], 1)
        self.assertEqual(status["pending_or_incomplete"], 2)
        self.assertEqual(status["status_counts"]["pending"], 2)

    def test_pid_status_reports_live_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runner.pid"
            path.write_text(str(os.getpid()))
            status = monitor.pid_status(path)

        self.assertTrue(status["exists"])
        self.assertTrue(status["alive"])
        self.assertEqual(status["pid"], os.getpid())

    def test_status_summary_marks_done_log_as_stale_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log = tmp_path / "run.log"
            log.write_text("[train] 200 epochs\n[epoch 199] loss=0.1 (1s)\n[done]\n")
            status_file = tmp_path / "status.json"
            status_file.write_text(json.dumps({
                "state": "running",
                "selected": 1,
                "running": [{"name": "unit", "log": str(log)}],
            }))

            old_root = monitor.ROOT
            try:
                monitor.ROOT = tmp_path
                status = monitor.status_summary(status_file)
            finally:
                monitor.ROOT = old_root

        self.assertEqual(status["state"], "stale_done")
        self.assertEqual(status["running"][0]["progress"], "final")
        self.assertEqual(status["running"][0]["eta"], "final")


if __name__ == "__main__":
    unittest.main()
