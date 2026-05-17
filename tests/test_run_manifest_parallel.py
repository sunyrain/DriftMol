import tempfile
import unittest
from pathlib import Path

import scripts.run_manifest_parallel as runner


class RunManifestParallelTest(unittest.TestCase):
    def test_rechecks_completion_before_launching_queued_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done_dir = root / "done"
            done_dir.mkdir()
            (done_dir / "final_metrics.json").write_text("{}")
            todo_dir = root / "todo"

            queue = [
                {"name": "done", "output_dir": str(done_dir)},
                {"name": "todo", "output_dir": str(todo_dir)},
            ]
            skipped = []

            entry = runner.pop_next_incomplete(queue, skipped)

        self.assertEqual(entry["name"], "todo")
        self.assertEqual([item["name"] for item in skipped], ["done"])
        self.assertEqual(queue, [])

    def test_force_keeps_completed_queued_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done_dir = root / "done"
            done_dir.mkdir()
            (done_dir / "final_metrics.json").write_text("{}")

            queue = [{"name": "done", "output_dir": str(done_dir)}]
            skipped = []

            entry = runner.pop_next_incomplete(queue, skipped, force=True)

        self.assertEqual(entry["name"], "done")
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
