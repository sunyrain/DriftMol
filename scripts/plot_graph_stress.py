#!/usr/bin/env python3
"""Plot the graph representation stress-test diagnostic figure."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GRAPH_JSON = ROOT / "results" / "graph_stress_test.json"
FIG_DIR = ROOT / "docs" / "figures"
STATUS_JSON = ROOT / "results" / "graph_stress_full_status.json"


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


QUALITY_ROWS = [
    ("vae_v2_kl01", "Graph VAE"),
    ("vae_v3_valence", "Valence VAE"),
    ("drifting_v2kl01_fix2", "Graph drift"),
    ("e36_dec_drift_cfg", "CFG graph drift"),
]

CONTROL_ROWS = [
    ("E30_phi_space_drift", "Phi drift", "#7A869A"),
    ("e36_dec_drift_cfg", "Graph QED", "#C65D32"),
    ("e40_logp_bins_queue", "Graph LogP", "#B58B2B"),
]


def load_payload() -> dict:
    return json.loads(GRAPH_JSON.read_text())


def row_by_run(payload: dict) -> dict[str, dict]:
    return {row["run"]: row for row in payload.get("rows", [])}


def savefig_atomic(fig: plt.Figure, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lstrip(".")
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    save_kwargs = dict(kwargs)
    if suffix == "pdf" and "metadata" not in save_kwargs:
        save_kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
    try:
        fig.savefig(tmp, format=suffix, **save_kwargs)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"failed to write non-empty figure: {tmp}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def plot() -> None:
    payload = load_payload()
    rows = row_by_run(payload)
    anchor = payload["anchor"]

    fig, (ax_q, ax_c) = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.8),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.34},
    )

    quality = []
    labels = []
    for run, label in QUALITY_ROWS:
        row = rows[run]
        quality.append([row["validity"], row["uniqueness"], row["novelty"]])
        labels.append(label)
    quality.append([anchor["validity"], anchor["uniqueness"], anchor["novelty"]])
    labels.append("SELFIES anchor")
    quality_arr = 100.0 * np.array(quality)

    y = np.arange(len(labels))
    colors = {"Validity": "#2A9D8F", "Uniqueness": "#D66A3A", "Novelty": "#4C72B0"}
    offsets = {"Validity": -0.18, "Uniqueness": 0.0, "Novelty": 0.18}
    for idx, metric in enumerate(["Validity", "Uniqueness", "Novelty"]):
        ax_q.scatter(
            quality_arr[:, idx],
            y + offsets[metric],
            s=30,
            color=colors[metric],
            edgecolor="white",
            linewidth=0.5,
            label=metric,
            zorder=3,
        )
    for yi in y:
        ax_q.plot([0, 103], [yi, yi], color="#EDF0F3", linewidth=0.7, zorder=0)
    ax_q.set_xlim(0, 103)
    ax_q.set_yticks(y)
    ax_q.set_yticklabels(labels)
    ax_q.invert_yaxis()
    ax_q.set_xlabel("Generated molecules (%)")
    ax_q.set_title("a. Generation quality", loc="left", fontweight="bold", pad=4)
    ax_q.grid(axis="x", color="#E5E9ED", linewidth=0.7)
    ax_q.spines["top"].set_visible(False)
    ax_q.spines["right"].set_visible(False)
    ax_q.legend(loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.2)

    for run, label, color in CONTROL_ROWS:
        row = rows[run]
        rho = row.get("control_best_rho")
        if rho is None:
            continue
        label_offsets = {
            "Phi drift": (6, 10),
            "Graph QED": (8, 12),
            "Graph LogP": (8, -16),
        }
        offset = label_offsets.get(label, (5, 4))
        ax_c.scatter(
            100.0 * row["uniqueness"],
            rho,
            s=44,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax_c.annotate(
            label,
            xy=(100.0 * row["uniqueness"], rho),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            color="#2F3A45",
        )

    ax_c.scatter(
        100.0 * anchor["uniqueness"],
        anchor["control_rho"],
        s=58,
        color="#2166AC",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax_c.annotate(
        "SELFIES",
        xy=(100.0 * anchor["uniqueness"], anchor["control_rho"]),
        xytext=(-45, 3),
        textcoords="offset points",
        fontsize=7,
        color="#1B4E82",
        fontweight="bold",
    )
    ax_c.axhline(anchor["control_rho"], color="#2166AC", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_c.set_xlim(15, 103)
    ax_c.set_ylim(-0.02, 0.55)
    ax_c.set_xlabel("Uniqueness (%)")
    ax_c.set_ylabel("Best control $\\rho$")
    ax_c.set_title("b. Control-diversity bottleneck", loc="left", fontweight="bold", pad=4)
    ax_c.grid(color="#E5E9ED", linewidth=0.7)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.86, wspace=0.34)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig_graph_bottleneck.pdf"
    png = FIG_DIR / "fig_graph_bottleneck.png"
    savefig_atomic(fig, pdf)
    savefig_atomic(fig, png, dpi=300)
    plt.close(fig)

    required_runs = {
        "e36_dec_drift_cfg_fresh",
        "e40_logp_bins_queue_fresh",
        "e36_no_drift_fresh",
    }
    available_runs = {row.get("run") for row in payload.get("rows", [])}
    missing_runs = sorted(required_runs - available_runs)
    raw_vs_repair_complete = bool(payload.get("raw_vs_repair"))
    complete = not missing_runs and raw_vs_repair_complete and pdf.exists() and png.exists()
    reason = (
        "fresh graph QED/LogP/no-drift, raw-vs-repaired validity, and bottleneck figure are complete"
        if complete
        else "missing fresh graph rows or raw-vs-repaired diagnostics"
    )
    STATUS_JSON.write_text(
        json.dumps(
            {
                "complete": complete,
                "reason": reason,
                "missing_runs": missing_runs,
                "raw_vs_repair_complete": raw_vs_repair_complete,
                "figure_pdf": str(pdf.relative_to(ROOT)),
                "figure_png": str(png.relative_to(ROOT)),
                "source": str(GRAPH_JSON.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    print(f"Wrote {STATUS_JSON}")


def main() -> None:
    plot()


if __name__ == "__main__":
    main()
