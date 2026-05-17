import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import scripts.audit_publication_completion as audit
from scripts.audit_publication_completion import (
    AUDIT_NAMES,
    QED_VARIANTS,
    REQUIRED_MANIFEST_NAMES,
    ZDIV_NAMES,
    check_artifacts,
    check_documentation_consistency,
    check_latex_table_sync,
    check_manuscript_bibliography,
    check_manuscript_and_figures,
    check_manuscript_build_prerequisites,
    check_manifest,
    check_experiment_groups,
    check_qed_seeds,
    check_tests,
)


class PublicationAuditTest(unittest.TestCase):
    def test_qed_seed_gate_requires_three_seeds_per_key_variant(self):
        status = {
            "qed_3seed": [
                {"variant": "A6", "n": 3},
                {"variant": "A8", "n": 3},
                {"variant": "F", "n": 2},
                {"variant": "G4", "n": 3},
            ]
        }
        checks = check_qed_seeds(status)
        self.assertFalse(checks[0].ok)
        self.assertIn("F=2/3", checks[0].evidence)

        status["qed_3seed"] = [{"variant": v, "n": 3} for v in QED_VARIANTS]
        checks = check_qed_seeds(status)
        self.assertTrue(checks[0].ok)

    def test_experiment_group_gate_requires_all_publication_groups(self):
        rows = [
            {"experiment": name, "status": "complete_pass"}
            for name in sorted(ZDIV_NAMES | AUDIT_NAMES | {"pub_G4_multi4_v2_s42"})
        ]
        checks = check_experiment_groups(rows)
        self.assertTrue(all(check.ok for check in checks))

        rows = [row for row in rows if row["experiment"] != "pub_G4_qed_zdiv2p0_s42"]
        checks = check_experiment_groups(rows)
        self.assertFalse(checks[0].ok)
        self.assertIn("pub_G4_qed_zdiv2p0_s42", checks[0].evidence)

        rows = [
            {"experiment": name, "status": "complete_pass"}
            for name in sorted(ZDIV_NAMES | AUDIT_NAMES | {"pub_G4_multi4_v2_s42"})
        ]
        for row in rows:
            if row["experiment"] == "pub_G4_multi4_v2_s42":
                row["status"] = "complete_fail"
        checks = check_experiment_groups(rows)
        self.assertFalse(checks[1].ok)
        self.assertIn("complete_fail", checks[1].evidence)

    def test_required_manifest_name_set_has_expected_size(self):
        self.assertEqual(len(REQUIRED_MANIFEST_NAMES), 24)

        checks = check_manifest({"pending_or_incomplete": 0})
        self.assertTrue(checks[0].ok)
        self.assertTrue(checks[1].ok)

    def test_manifest_gate_requires_existing_configs_and_matching_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            entries = []
            for name in sorted(REQUIRED_MANIFEST_NAMES):
                config = tmp_path / f"{name}.yaml"
                if name != "pub_F_qed_s42":
                    config.write_text("seed: 42\n")
                entries.append(
                    {
                        "name": name,
                        "config": str(config),
                        "command": f"python -m src.train.train_selfies_cfg --config {config}",
                    }
                )
            entries[1]["command"] = "python -m src.train.train_selfies_cfg --config wrong.yaml"
            manifest = tmp_path / "manifest.json"
            manifest.write_text(json.dumps({"entries": entries}))

            old_manifest = audit.MANIFEST
            try:
                audit.MANIFEST = manifest
                checks = check_manifest({"pending_or_incomplete": 0})
            finally:
                audit.MANIFEST = old_manifest

            self.assertTrue(checks[0].ok)
            self.assertFalse(checks[1].ok)
            self.assertIn("missing configs:", checks[1].evidence)
            self.assertIn("command mismatches:", checks[1].evidence)

    def test_artifact_gate_rejects_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results = tmp_path / "results"
            tables = results / "tables"
            tables.mkdir(parents=True)
            for rel in [
                "publication_summary.md",
                "publication_results.csv",
                "publication_status.json",
                "tables/tab_qed_main.tex",
                "tables/tab_multi4_v2.tex",
                "tables/tab_qed_3seed.tex",
            ]:
                (results / rel).write_text("x")
            (tables / "tab_qed_main.tex").write_text("")

            old_results = audit.RESULTS
            try:
                audit.RESULTS = results
                checks = check_artifacts({"pending_or_incomplete": 0})
            finally:
                audit.RESULTS = old_results

            self.assertFalse(checks[0].ok)
            self.assertIn("tab_qed_main.tex=0B", checks[0].evidence)

    def test_latex_table_sync_rejects_seed_replicate_pollution(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tables = tmp_path / "tables"
            tables.mkdir()
            rows = [
                {
                    "root": "final",
                    "manifest_group": "",
                    "variant": "F",
                    "condition": "qed",
                    "status": "complete_pass",
                    "alpha": "alpha=5.0",
                    "spearman_rho": "0.49337",
                    "uniqueness": "0.9465",
                    "mae": "0.200",
                },
                {
                    "root": "seeds",
                    "manifest_group": "qed_3seed",
                    "variant": "F",
                    "condition": "qed",
                    "status": "complete_pass",
                    "alpha": "alpha=5.0",
                    "spearman_rho": "0.52361",
                    "uniqueness": "0.9429",
                    "mae": "0.209",
                },
            ]
            status = {"qed_3seed": []}
            (tables / "tab_qed_main.tex").write_text(
                "\\begin{tabular}{l c c c c}\n"
                "\\textbf{Full} & 5.0 & 0.524 & 94.3 & 0.209 \\\\\n"
                "\\end{tabular}\n"
            )
            (tables / "tab_multi4_v2.tex").write_text("\\begin{tabular}{l}\\end{tabular}\n")
            (tables / "tab_qed_3seed.tex").write_text("\\begin{tabular}{l}\\end{tabular}\n")

            old_results = audit.RESULTS
            try:
                audit.RESULTS = tmp_path
                checks = check_latex_table_sync(rows, status)
            finally:
                audit.RESULTS = old_results

            self.assertFalse(checks[0].ok)
            self.assertIn("tab_qed_main.tex missing 1 expected rows", checks[0].evidence)
            self.assertIn("tab_qed_main.tex has 1 unexpected rows", checks[0].evidence)

    def test_figure_gate_rejects_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            figures = tmp_path / "figures"
            figures.mkdir()
            paper = tmp_path / "paper.md"
            stems = [
                "fig1_main",
                "fig2_qed_ablation",
                "fig3_multi4_v2",
                "fig4_qed_seed_ci",
                "fig5_zdiv_pareto",
            ]
            paper.write_text("\n".join(f"{stem}.pdf" for stem in stems))
            for stem in stems:
                (figures / f"{stem}.pdf").write_bytes(b"x" * 2048)
                (figures / f"{stem}.png").write_bytes(b"x" * 2048)
            (figures / "fig4_qed_seed_ci.png").write_bytes(b"x")

            old_paper, old_figures = audit.PAPER, audit.FIGURES
            try:
                audit.PAPER = paper
                audit.FIGURES = figures
                checks = check_manuscript_and_figures()
            finally:
                audit.PAPER = old_paper
                audit.FIGURES = old_figures

            self.assertFalse(checks[1].ok)
            self.assertIn("invalid: fig4_qed_seed_ci.png=1B", checks[1].evidence)

    def test_manuscript_gate_requires_table_labels_and_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            figures = tmp_path / "figures"
            figures.mkdir()
            paper = tmp_path / "paper.md"
            stems = [
                "fig1_main",
                "fig2_qed_ablation",
                "fig3_multi4_v2",
                "fig4_qed_seed_ci",
                "fig5_zdiv_pareto",
            ]
            for stem in stems:
                (figures / f"{stem}.pdf").write_bytes(b"x" * 2048)
                (figures / f"{stem}.png").write_bytes(b"x" * 2048)
            paper.write_text(
                "\\label{tab:qed_main} Table~\\ref{tab:qed_main}\n"
                "\\label{tab:multi4} Table~\\ref{tab:multi4}\n"
                "\\label{tab:qed_3seed}\n"
                + "\n".join(f"{stem}.pdf" for stem in stems)
            )

            old_paper, old_figures = audit.PAPER, audit.FIGURES
            try:
                audit.PAPER = paper
                audit.FIGURES = figures
                checks = check_manuscript_and_figures()
            finally:
                audit.PAPER = old_paper
                audit.FIGURES = old_figures

            self.assertFalse(checks[0].ok)
            self.assertIn("missing refs: tab:qed_3seed", checks[0].evidence)

    def test_manuscript_gate_requires_figure_includes_labels_and_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            figures = tmp_path / "figures"
            figures.mkdir()
            paper = tmp_path / "paper.md"
            fig_labels = {
                "fig1_main": "fig:main",
                "fig2_qed_ablation": "fig:qed_ablation",
                "fig3_multi4_v2": "fig:multi4_v2",
                "fig4_qed_seed_ci": "fig:qed_seed_ci",
                "fig5_zdiv_pareto": "fig:zdiv_pareto",
            }
            for stem in fig_labels:
                (figures / f"{stem}.pdf").write_bytes(b"x" * 2048)
                (figures / f"{stem}.png").write_bytes(b"x" * 2048)
            table_text = "\n".join(
                f"\\label{{{label}}} Table~\\ref{{{label}}}"
                for label in ["tab:qed_main", "tab:multi4", "tab:qed_3seed"]
            )
            figure_text = "\n".join(
                f"{stem}.pdf \\label{{{label}}} Figure~\\ref{{{label}}}"
                for stem, label in fig_labels.items()
                if stem != "fig5_zdiv_pareto"
            )
            paper.write_text(table_text + "\n" + figure_text + "\nfig5_zdiv_pareto.pdf\n")

            old_paper, old_figures = audit.PAPER, audit.FIGURES
            try:
                audit.PAPER = paper
                audit.FIGURES = figures
                checks = check_manuscript_and_figures()
            finally:
                audit.PAPER = old_paper
                audit.FIGURES = old_figures

            self.assertFalse(checks[2].ok)
            self.assertIn("fig:zdiv_pareto label", checks[2].evidence)
            self.assertIn("fig:zdiv_pareto ref", checks[2].evidence)

    def test_aaai_source_gate_requires_current_labels_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paper = tmp_path / "paper.tex"
            paper.write_text(
                "\\documentclass[letterpaper]{article}\n"
                "\\usepackage[draft]{aaai2026}\n"
                "\\title{DriftingMol: Decoder-Coupled Drift for One-Pass Property-Conditional Molecular Generation}\n"
                "\\author{Jiangjie Qiu, Yijun Li, Wentao Li, Xiaonan Wang\\thanks{Corresponding author.}}\n"
                "\\affiliations{Beijing Key Laboratory of Artificial Intelligence for Advanced Chemical Engineering Materials}\n"
                "\\begin{document}\n"
                "\\label{tab:qed-main} Table~\\ref{tab:qed-main}\n"
                "\\label{tab:qed-diversity} Table~\\ref{tab:qed-diversity}\n"
                "\\label{tab:qed-3seed} Table~\\ref{tab:qed-3seed}\n"
                "\\label{tab:multi4} Table~\\ref{tab:multi4}\n"
                "fig1_main.pdf \\label{fig:main} Figure~\\ref{fig:main}\n"
                "fig2_qed_ablation.pdf \\label{fig:qed-ablation} Figure~\\ref{fig:qed-ablation}\n"
                "fig3_multi4_v2.pdf \\label{fig:multi4} Figure~\\ref{fig:multi4}\n"
                "fig4_qed_seed_ci.pdf \\label{fig:qed-seed-ci} Figure~\\ref{fig:qed-seed-ci}\n"
                "fig5_zdiv_pareto.pdf \\label{fig:zdiv-pareto} Figure~\\ref{fig:zdiv-pareto}\n"
                "\\bibliography{docs/references_aaai}\n"
                "\\end{document}\n"
            )

            old_paper = audit.AAAI_PAPER
            try:
                audit.AAAI_PAPER = paper
                meta_checks = audit.check_aaai_source_metadata()
                fig_checks = audit.check_aaai_manuscript_and_figures()
            finally:
                audit.AAAI_PAPER = old_paper

            self.assertTrue(meta_checks[0].ok)
            self.assertTrue(fig_checks[0].ok)
            self.assertTrue(fig_checks[1].ok)

    def test_manuscript_build_gate_requires_engine_class_and_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paper = tmp_path / "paper.md"
            build = tmp_path / "PAPER_BUILD.md"
            paper.write_text("\\documentclass[pdflatex,sn-nature]{sn-jnl}\n")
            build.write_text("pdflatex docs/PAPER_DRAFT.md\n")

            old_paper, old_build, old_root = audit.PAPER, audit.PAPER_BUILD, audit.ROOT
            try:
                audit.PAPER = paper
                audit.PAPER_BUILD = build
                audit.ROOT = tmp_path
                with patch("scripts.audit_publication_completion.shutil.which", return_value=None):
                    checks = check_manuscript_build_prerequisites()

                self.assertFalse(checks[0].ok)
                self.assertIn("TeX engine", checks[0].evidence)
                self.assertIn("sn-jnl.cls", checks[0].evidence)

                (tmp_path / "sn-jnl.cls").write_text("% test class\n")
                with patch("scripts.audit_publication_completion.shutil.which", return_value="/usr/bin/pdflatex"):
                    checks = check_manuscript_build_prerequisites()
            finally:
                audit.PAPER = old_paper
                audit.PAPER_BUILD = old_build
                audit.ROOT = old_root

            self.assertTrue(checks[0].ok)

    def test_manuscript_build_gate_requires_texinputs_for_docs_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs = tmp_path / "docs"
            docs.mkdir()
            paper = docs / "paper.md"
            build = tmp_path / "PAPER_BUILD.md"
            paper.write_text("\\documentclass[pdflatex,sn-nature]{sn-jnl}\n")
            (docs / "sn-jnl.cls").write_text("% test class\n")
            build.write_text("pdflatex docs/paper.md\n")

            old_paper, old_build, old_root = audit.PAPER, audit.PAPER_BUILD, audit.ROOT
            try:
                audit.PAPER = paper
                audit.PAPER_BUILD = build
                audit.ROOT = tmp_path
                with patch("scripts.audit_publication_completion.shutil.which", return_value="/usr/bin/pdflatex"):
                    checks = check_manuscript_build_prerequisites()

                self.assertFalse(checks[0].ok)
                self.assertIn("TEXINPUTS=docs//:", checks[0].evidence)

                build.write_text("TEXINPUTS=docs//: pdflatex docs/paper.md\n")
                with patch("scripts.audit_publication_completion.shutil.which", return_value="/usr/bin/pdflatex"):
                    checks = check_manuscript_build_prerequisites()
            finally:
                audit.PAPER = old_paper
                audit.PAPER_BUILD = old_build
                audit.ROOT = old_root

            self.assertTrue(checks[0].ok)

    def test_documentation_gate_rejects_stale_protocol_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            readme = tmp_path / "README.md"
            full_results = tmp_path / "FULL_RESULTS.md"
            readme.write_text(
                "See results/publication_summary.md\n"
                "| Multi-4-property v2 | Single-tau | rho = 0.598 |\n"
            )
            full_results.write_text(
                "# DriftingMol - Legacy Static Snapshot\n"
                "Table 4b: Fair Multi-property v2\n"
            )

            old_readme, old_full_results = audit.README, audit.FULL_RESULTS
            try:
                audit.README = readme
                audit.FULL_RESULTS = full_results
                checks = check_documentation_consistency()
                self.assertTrue(checks[0].ok)

                full_results.write_text(
                    "# DriftingMol - Legacy Static Snapshot\n"
                    "Table 4b: Fair Multi-property v2\n"
                    "Single temperature τ=10 instead of multi-scale {1,5,10,20}\n"
                )
                checks = check_documentation_consistency()
            finally:
                audit.README = old_readme
                audit.FULL_RESULTS = old_full_results

            self.assertFalse(checks[0].ok)
            self.assertIn("old_temperature_tau_10", checks[0].evidence)

    def test_bibliography_gate_requires_resolved_and_used_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "paper.md"
            paper.write_text(
                "Prior work~\\cite{A,B}.\n"
                "\\begin{thebibliography}{2}\n"
                "\\bibitem{A} A.\n"
                "\\bibitem{C} C.\n"
                "\\end{thebibliography}\n"
            )
            old_paper = audit.PAPER
            try:
                audit.PAPER = paper
                checks = check_manuscript_bibliography()
            finally:
                audit.PAPER = old_paper

            self.assertFalse(checks[0].ok)
            self.assertIn("missing bibitems: B", checks[0].evidence)
            self.assertIn("unused bibitems: C", checks[0].evidence)

    def test_test_gate_reports_test_count_with_noisy_stdout(self):
        class Proc:
            returncode = 0
            stdout = "Saved: /tmp/fig.png\nRan 31 tests in 0.1s\n\nOK\nSaved: /tmp/other.png\n"

        with patch("scripts.audit_publication_completion.subprocess.run", return_value=Proc()):
            checks = check_tests(True)

        self.assertTrue(checks[0].ok)
        self.assertEqual(checks[0].evidence, "OK (31 tests)")


if __name__ == "__main__":
    unittest.main()
