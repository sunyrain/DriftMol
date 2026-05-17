import tempfile
import unittest
from pathlib import Path

from scripts.render_faithful_supplement import build_standalone, inline_inputs


class RenderFaithfulSupplementTest(unittest.TestCase):
    def test_inline_inputs_replaces_table_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "docs"
            table = root / "results/tables/unit.tex"
            table.parent.mkdir(parents=True)
            table.write_text("\\begin{tabular}{c}\nA\\\\\n\\end{tabular}\n")

            text = "before\n  \\input{../results/tables/unit.tex}\nafter\n"
            rendered = inline_inputs(text, base)

        self.assertIn("% BEGIN inlined", rendered)
        self.assertIn("\\begin{tabular}{c}", rendered)
        self.assertIn("  A\\\\", rendered)
        self.assertNotIn("\\input{", rendered)

    def test_build_standalone_wraps_inlined_section_as_compileable_source(self):
        text = "\\section{Faithful Drifting Reproduction}\nBody without inputs.\n"
        rendered = build_standalone(text)

        self.assertIn("\\documentclass[letterpaper]{article}", rendered)
        self.assertIn("\\usepackage[draft]{aaai2026}", rendered)
        self.assertIn("\\maketitle", rendered)
        self.assertIn("\\section{Faithful Drifting Reproduction}", rendered)
        self.assertTrue(rendered.rstrip().endswith("\\end{document}"))
        self.assertNotIn("\\input{", rendered)


if __name__ == "__main__":
    unittest.main()
