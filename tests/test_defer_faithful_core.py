import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.defer_faithful_core_after_destructive as defer


class DeferFaithfulCoreTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def _argv(self, watch: Path, faithful: Path, log_dir: Path, pid_file: Path | None = None):
        pid_file = pid_file or (watch.parent / "defer.pid")
        return [
            "defer_faithful_core_after_destructive.py",
            "--watch-status",
            str(watch),
            "--faithful-status",
            str(faithful),
            "--devices",
            "0,2,3",
            "--poll-seconds",
            "0",
            "--log-dir",
            str(log_dir),
            "--pid-file",
            str(pid_file),
        ]

    def test_launches_faithful_core_after_destructive_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watch = root / "destructive_status.json"
            faithful = root / "faithful_status.json"
            log_dir = root / "logs"
            self._write_json(watch, {"state": "completed", "running": [], "queued": []})

            with patch.object(sys, "argv", self._argv(watch, faithful, log_dir)), patch(
                "scripts.defer_faithful_core_after_destructive.subprocess.run"
            ) as refresh, patch(
                "scripts.defer_faithful_core_after_destructive.subprocess.call", return_value=0
            ) as launch:
                rc = defer.main()
            pid_text = (watch.parent / "defer.pid").read_text().strip()

        self.assertEqual(rc, 0)
        self.assertEqual(pid_text, str(os.getpid()))
        self.assertEqual(refresh.call_count, 4)
        launch.assert_called_once()
        cmd = launch.call_args.args[0]
        self.assertIn("scripts/run_manifest_parallel.py", cmd)
        self.assertIn("--group", cmd)
        self.assertIn("faithful_core", cmd)
        self.assertIn("--devices", cmd)
        self.assertIn("0,2,3", cmd)
        refresh_cmds = [call.args[0] for call in refresh.call_args_list]
        self.assertIn([sys.executable, "scripts/collect_extension_results.py"], refresh_cmds)
        self.assertIn([sys.executable, "scripts/collect_faithful_drifting_results.py"], refresh_cmds)
        self.assertIn([sys.executable, "scripts/audit_drifting_faithfulness.py"], refresh_cmds)
        self.assertIn([sys.executable, "scripts/audit_reviewer_experiment_readiness.py"], refresh_cmds)

    def test_does_not_launch_when_destructive_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watch = root / "destructive_status.json"
            faithful = root / "faithful_status.json"
            log_dir = root / "logs"
            self._write_json(watch, {"state": "failed", "failures": ["unit"]})

            with patch.object(sys, "argv", self._argv(watch, faithful, log_dir)), patch(
                "scripts.defer_faithful_core_after_destructive.subprocess.call"
            ) as launch:
                rc = defer.main()

        self.assertEqual(rc, 2)
        launch.assert_not_called()

    def test_exits_if_faithful_core_already_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watch = root / "destructive_status.json"
            faithful = root / "faithful_status.json"
            log_dir = root / "logs"
            self._write_json(watch, {"state": "completed"})
            self._write_json(faithful, {"state": "running"})

            with patch.object(sys, "argv", self._argv(watch, faithful, log_dir)), patch(
                "scripts.defer_faithful_core_after_destructive.subprocess.call"
            ) as launch:
                rc = defer.main()

        self.assertEqual(rc, 0)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
