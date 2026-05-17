#!/usr/bin/env python3
"""Render the faithful-Drifting supplement with generated tables inlined."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "SUPPLEMENT_FAITHFUL_DRIFTING.tex"
OUT = ROOT / "docs" / "SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex"
STANDALONE_OUT = ROOT / "docs" / "SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex"
INPUT_RE = re.compile(r"^(?P<indent>\s*)\\input\{(?P<path>[^}]+)\}\s*$")

STANDALONE_PREAMBLE = r"""\documentclass[letterpaper]{article}

\usepackage[draft]{aaai2026}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}

\urlstyle{rm}
\def\UrlFont{\rm}
\frenchspacing
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}
\setcounter{secnumdepth}{2}

\title{Supplementary Material: Faithful Drifting Reproduction for DriftingMol}
\author{Jiangjie Qiu, Yijun Li, Wentao Li, Xiaonan Wang\thanks{Corresponding author.}}
\affiliations{Beijing Key Laboratory of Artificial Intelligence for Advanced Chemical Engineering Materials}

\begin{document}
\maketitle
"""


def inline_inputs(text: str, base_dir: Path) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        match = INPUT_RE.match(line)
        if not match:
            lines.append(line)
            continue

        input_path = (base_dir / match.group("path")).resolve()
        rel = input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path
        table = input_path.read_text().rstrip()
        indent = match.group("indent")
        lines.append(f"{indent}% BEGIN inlined {rel}")
        lines.extend(f"{indent}{table_line}" if table_line else "" for table_line in table.splitlines())
        lines.append(f"{indent}% END inlined {rel}")
    return "\n".join(lines) + "\n"


def build_standalone(section_text: str) -> str:
    return STANDALONE_PREAMBLE + "\n" + section_text.strip() + "\n\n\\end{document}\n"


def render(
    template: Path = TEMPLATE,
    out: Path = OUT,
    standalone_out: Path = STANDALONE_OUT,
) -> None:
    rendered = inline_inputs(template.read_text(), template.parent)
    out.write_text(rendered)
    standalone_out.write_text(build_standalone(rendered))


def main() -> int:
    render()
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {STANDALONE_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
