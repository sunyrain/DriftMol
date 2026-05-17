import unittest

from scripts.finalize_publication_when_ready import (
    GpuStat,
    choose_idle_gpu,
    experiments_complete,
    parse_gpu_stats,
    pending_count,
    qed_seed_summary,
    refresh_commands,
)


class FinalizePublicationTest(unittest.TestCase):
    def test_parse_gpu_stats_ignores_bad_rows(self):
        text = "0, 100, 8320\n1, 3, 812\nbad row\n2, 0, 512\n"
        self.assertEqual(
            parse_gpu_stats(text),
            [GpuStat(0, 100, 8320), GpuStat(1, 3, 812), GpuStat(2, 0, 512)],
        )

    def test_choose_idle_gpu_prefers_low_memory_then_util(self):
        stats = [
            GpuStat(0, 100, 8320),
            GpuStat(1, 5, 1200),
            GpuStat(2, 0, 900),
        ]
        self.assertEqual(
            choose_idle_gpu(stats, max_util_pct=10, max_memory_mb=2000),
            GpuStat(2, 0, 900),
        )
        self.assertIsNone(choose_idle_gpu(stats, max_util_pct=0, max_memory_mb=800))

    def test_completion_and_summary_helpers(self):
        self.assertIsNone(pending_count({}))
        self.assertFalse(experiments_complete({"pending_or_incomplete": 1}))
        self.assertTrue(experiments_complete({"pending_or_incomplete": "0"}))
        self.assertEqual(
            qed_seed_summary({"qed_3seed": [{"variant": "F", "n": 3}, {"variant": "A6", "n": 2}]}),
            "A6=2/3, F=3/3",
        )

    def test_refresh_commands_include_all_figure_generators(self):
        command_text = [" ".join(cmd) for cmd in refresh_commands()]
        self.assertTrue(any("scripts/plot_main_figure.py" in cmd for cmd in command_text))
        self.assertTrue(any("scripts/plot_result_figures.py" in cmd for cmd in command_text))


if __name__ == "__main__":
    unittest.main()
