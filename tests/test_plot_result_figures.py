import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import scripts.plot_result_figures as plot


def _qed_row(variant: str, seed: int, rho: float, root: str = "seeds") -> dict[str, str]:
    return {
        "condition": "qed",
        "status": "complete_pass",
        "variant": variant,
        "seed": str(seed),
        "root": root,
        "manifest_group": "qed_3seed" if root == "seeds" else "",
        "spearman_rho": str(rho),
    }


class PlotResultFiguresTest(unittest.TestCase):
    def quiet_call(self, func, *args):
        with redirect_stdout(io.StringIO()):
            return func(*args)

    def test_qed_seed_ci_removes_stale_until_all_key_variants_have_three_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig_dir = Path(tmp)
            for suffix in ("pdf", "png"):
                (fig_dir / f"fig4_qed_seed_ci.{suffix}").write_bytes(b"stale")

            old_fig_dir = plot.FIG_DIR
            try:
                plot.FIG_DIR = fig_dir
                rows = [
                    _qed_row("A6", 42, 0.50, root="final"),
                    _qed_row("A6", 43, 0.51),
                    _qed_row("A8", 42, 0.50, root="final"),
                ]
                self.quiet_call(plot.plot_qed_seed_ci, rows)
            finally:
                plot.FIG_DIR = old_fig_dir

            self.assertFalse((fig_dir / "fig4_qed_seed_ci.pdf").exists())
            self.assertFalse((fig_dir / "fig4_qed_seed_ci.png").exists())

    def test_main_qed_rows_ignore_publication_seed_replicates(self):
        rows = [
            {
                "root": "final",
                "manifest_group": "",
                "variant": "F",
                "condition": "qed",
                "status": "complete_pass",
                "spearman_rho": "0.493",
            },
            {
                "root": "seeds",
                "manifest_group": "qed_3seed",
                "variant": "F",
                "condition": "qed",
                "status": "complete_pass",
                "spearman_rho": "0.524",
            },
        ]

        selected = plot.main_qed_rows(rows)

        self.assertEqual(selected["F"]["spearman_rho"], "0.493")

    def test_qed_seed_ci_generates_when_all_key_variants_have_three_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig_dir = Path(tmp)
            rows = []
            for variant, base in [("A8", 0.51), ("A6", 0.50), ("F", 0.49), ("G4", 0.44)]:
                rows.extend(
                    [
                        _qed_row(variant, 42, base, root="final"),
                        _qed_row(variant, 43, base + 0.01),
                        _qed_row(variant, 44, base - 0.01),
                    ]
                )

            old_fig_dir = plot.FIG_DIR
            try:
                plot.FIG_DIR = fig_dir
                self.quiet_call(plot.plot_qed_seed_ci, rows)
            finally:
                plot.FIG_DIR = old_fig_dir

            self.assertGreater((fig_dir / "fig4_qed_seed_ci.pdf").stat().st_size, 1024)
            self.assertGreater((fig_dir / "fig4_qed_seed_ci.png").stat().st_size, 1024)

    def test_zdiv_pareto_removes_stale_until_two_points_and_then_generates(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig_dir = Path(tmp)
            for suffix in ("pdf", "png"):
                (fig_dir / f"fig5_zdiv_pareto.{suffix}").write_bytes(b"stale")

            old_fig_dir = plot.FIG_DIR
            try:
                plot.FIG_DIR = fig_dir
                self.quiet_call(
                    plot.plot_zdiv_pareto,
                    [
                        {
                            "root": "zdiv",
                            "status": "complete_pass",
                            "experiment": "pub_G4_qed_zdiv0p5_s42",
                            "spearman_rho": "0.45",
                            "uniqueness": "0.95",
                        }
                    ]
                )
                self.assertFalse((fig_dir / "fig5_zdiv_pareto.pdf").exists())
                self.assertFalse((fig_dir / "fig5_zdiv_pareto.png").exists())

                self.quiet_call(
                    plot.plot_zdiv_pareto,
                    [
                        {
                            "root": "zdiv",
                            "status": "complete_pass",
                            "experiment": "pub_G4_qed_zdiv0p5_s42",
                            "spearman_rho": "0.45",
                            "uniqueness": "0.95",
                        },
                        {
                            "root": "zdiv",
                            "status": "complete_pass",
                            "experiment": "pub_G4_qed_zdiv2p0_s42",
                            "spearman_rho": "0.42",
                            "uniqueness": "0.98",
                        },
                    ]
                )
            finally:
                plot.FIG_DIR = old_fig_dir

            self.assertGreater((fig_dir / "fig5_zdiv_pareto.pdf").stat().st_size, 1024)
            self.assertGreater((fig_dir / "fig5_zdiv_pareto.png").stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()
