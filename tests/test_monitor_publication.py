import json
import os
import tempfile
import unittest
from pathlib import Path

import scripts.monitor_publication as monitor
from scripts.monitor_publication import format_age, format_eta


class MonitorPublicationTest(unittest.TestCase):
    def test_format_eta_from_epoch_duration(self):
        self.assertEqual(format_eta(10, 20, "[epoch 10] loss=1 (60s)", False), "10m")
        self.assertEqual(format_eta(10, 20, "[epoch 10] loss=1 (360s)", False), "1h00m")

    def test_format_eta_final_or_missing(self):
        self.assertEqual(format_eta(300, 300, "[final] Loading best checkpoint...", False), "final")
        self.assertEqual(format_eta(10, 20, "[epoch 10] loss=1", False), "-")
        self.assertEqual(format_eta(10, 20, "[epoch 10] loss=1 (60s)", True), "-")

    def test_format_age(self):
        self.assertEqual(format_age(None), "-")
        self.assertEqual(format_age(9), "9s")
        self.assertEqual(format_age(90), "1m")
        self.assertEqual(format_age(3660), "1h01m")

    def test_status_rows_skips_empty_completed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pub = tmp_path / "outputs" / "publication"
            pub.mkdir(parents=True)
            (pub / "runner_status_empty.json").write_text(json.dumps({
                "state": "completed",
                "updated_at": "2026-05-08T00:00:00+00:00",
            }))
            old_root, old_pub, old_log_dir = monitor.ROOT, monitor.PUB, monitor.LOG_DIR
            try:
                monitor.ROOT = tmp_path
                monitor.PUB = pub
                monitor.LOG_DIR = pub / "logs"
                self.assertEqual(monitor.status_rows(), [])
            finally:
                monitor.ROOT = old_root
                monitor.PUB = old_pub
                monitor.LOG_DIR = old_log_dir

    def test_watcher_rows_reports_pid_liveness(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pub = tmp_path / "outputs" / "publication"
            pub.mkdir(parents=True)
            (pub / "runner_followup_gpu0.pid").write_text(str(os.getpid()))
            (pub / "runner_status_followup_gpu0.json").write_text(json.dumps({
                "state": "running",
                "entry": {"name": "pub_unit"},
            }))
            old_root, old_pub = monitor.ROOT, monitor.PUB
            try:
                monitor.ROOT = tmp_path
                monitor.PUB = pub
                rows = monitor.watcher_rows()
            finally:
                monitor.ROOT = old_root
                monitor.PUB = old_pub

            self.assertEqual(rows[0]["gpu"], "0")
            self.assertEqual(rows[0]["alive"], "yes")
            self.assertEqual(rows[0]["state"], "running")
            self.assertEqual(rows[0]["name"], "pub_unit")
            self.assertEqual(rows[0]["status_exists"], "yes")
            self.assertEqual(rows[0]["log_exists"], "no")

    def test_watcher_rows_reports_missing_pending_status_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pub = tmp_path / "outputs" / "publication"
            pub.mkdir(parents=True)
            (pub / "runner_followup_gpu1.pid").write_text(str(os.getpid()))
            old_root, old_pub = monitor.ROOT, monitor.PUB
            try:
                monitor.ROOT = tmp_path
                monitor.PUB = pub
                rows = monitor.watcher_rows()
            finally:
                monitor.ROOT = old_root
                monitor.PUB = old_pub

            self.assertEqual(rows[0]["gpu"], "1")
            self.assertEqual(rows[0]["state"], "waiting")
            self.assertEqual(rows[0]["status_exists"], "no")
            self.assertEqual(rows[0]["log_exists"], "no")


if __name__ == "__main__":
    unittest.main()
