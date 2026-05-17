import tempfile
import unittest
from pathlib import Path

import scripts.report_checkpoint_cleanup_candidates as report


class CheckpointCleanupCandidatesTest(unittest.TestCase):
    def test_find_candidates_requires_final_metrics_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "outputs/complete_run"
            complete.mkdir(parents=True)
            (complete / "last.pt").write_bytes(b"x" * 10)
            (complete / "best.pt").write_bytes(b"best")
            (complete / "final_metrics.json").write_text("{}\n")

            active = root / "outputs/active_run"
            active.mkdir(parents=True)
            (active / "last.pt").write_bytes(b"x" * 20)
            (active / "best.pt").write_bytes(b"best")

            no_best = root / "outputs/no_best_run"
            no_best.mkdir(parents=True)
            (no_best / "last.pt").write_bytes(b"x" * 30)
            (no_best / "final_metrics.json").write_text("{}\n")

            candidates = report.find_candidates(root)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][1].name, "last.pt")
        self.assertEqual(candidates[0][1].parent.name, "complete_run")

    def test_find_candidates_excludes_active_output_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "outputs/complete_run"
            complete.mkdir(parents=True)
            (complete / "last.pt").write_bytes(b"x" * 10)
            (complete / "best.pt").write_bytes(b"best")
            (complete / "final_metrics.json").write_text("{}\n")

            candidates = report.find_candidates(
                root,
                active_run_dirs={complete.resolve(strict=False)},
            )

        self.assertEqual(candidates, [])

    def test_find_candidates_requires_valid_final_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "outputs/bad_final"
            run.mkdir(parents=True)
            (run / "last.pt").write_bytes(b"x" * 10)
            (run / "best.pt").write_bytes(b"best")
            (run / "final_metrics.json").write_text("{not-json}\n")

            candidates = report.find_candidates(root)

        self.assertEqual(candidates, [])

    def test_delete_candidates_rechecks_safety_and_keeps_best_metrics_and_active_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "outputs/complete_run"
            complete.mkdir(parents=True)
            (complete / "last.pt").write_bytes(b"x" * 10)
            (complete / "best.pt").write_bytes(b"best")
            (complete / "final_metrics.json").write_text("{}\n")

            active = root / "outputs/active_run"
            active.mkdir(parents=True)
            (active / "last.pt").write_bytes(b"x" * 20)
            (active / "best.pt").write_bytes(b"best")
            (active / "final_metrics.json").write_text("{}\n")

            active_dirs = {active.resolve(strict=False)}
            deleted = report.delete_candidates(
                [
                    ((complete / "last.pt").stat().st_size, complete / "last.pt"),
                    ((active / "last.pt").stat().st_size, active / "last.pt"),
                ],
                root.resolve(strict=False),
                active_dirs,
            )

            self.assertEqual([path.parent.name for _, path in deleted], ["complete_run"])
            self.assertFalse((complete / "last.pt").exists())
            self.assertTrue((complete / "best.pt").exists())
            self.assertTrue((complete / "final_metrics.json").exists())
            self.assertTrue((active / "last.pt").exists())

    def test_extract_config_paths_supports_space_and_equals_forms(self):
        command = (
            "python -m src.train.train_selfies_cfg --config configs/a.yaml "
            "python -m src.train.train_selfies_cfg --config=configs/b.yaml"
        )

        paths = report.extract_config_paths(command)

        self.assertEqual(paths, [Path("configs/a.yaml"), Path("configs/b.yaml")])

    def test_load_output_dir_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "configs/run.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("experiment:\n  output_dir: outputs/run\n")

            output_dir = report.load_output_dir_from_config(config, root)

        self.assertEqual(output_dir, (root / "outputs/run").resolve(strict=False))

    def test_write_delete_log_appends_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = root / "outputs/run/last.pt"
            last.parent.mkdir(parents=True)
            last.write_bytes(b"x" * 10)
            out = root / "deleted.md"

            report.write_delete_log(out, [(10, last)], active_run_dirs=set())
            report.write_delete_log(out, [(10, last)], active_run_dirs=set())

            text = out.read_text()
        self.assertEqual(text.count("## Deletion Batch"), 2)
        self.assertIn("Deleted files in batch: 1", text)

    def test_write_report_is_read_only_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = root / "outputs/run/last.pt"
            last.parent.mkdir(parents=True)
            last.write_bytes(b"x" * 1024)
            out = root / "report.md"

            report.write_report(out, [(last.stat().st_size, last)], limit=10)

            text = out.read_text()
        self.assertIn("no files were deleted", text)
        self.assertIn("1.0K", text)
        self.assertIn("last.pt", text)


if __name__ == "__main__":
    unittest.main()
