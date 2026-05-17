#!/usr/bin/env python3
"""Generate result figures for the DriftingMol manuscript.

Outputs:
  docs/figures/fig2_qed_ablation.{pdf,png}
  docs/figures/fig3_multi4_v2.{pdf,png}
  docs/figures/fig4_qed_seed_ci.{pdf,png} when all key QED variants have 3 seeds
  docs/figures/fig5_zdiv_pareto.{pdf,png} when at least two z-div points are complete
"""
from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "publication_results.csv"
FIG_DIR = ROOT / "docs" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

QED_ORDER = [
    "A8", "A6", "F", "A1", "C2", "C3", "C1", "C5",
    "A2", "C4", "B1", "A3", "A4", "B3", "B2",
]
QED_LABELS = {
    "A8": "Decoder coupling (no diversity)",
    "A6": "Decoder coupling (single $\\tau$)",
    "F": "DriftingMol",
    "A1": "Property-head baseline",
    "C2": "LatentMAE-guided drift",
    "C3": "LatentMAE + latent drift",
    "C1": "LatentMAE drift",
    "C5": "LatentMAE-guided + decoder",
    "A2": "Latent-space drift",
    "C4": "LatentMAE + decoder",
    "B1": "Random-feature drift control",
    "A3": "Detached decoder-feature control",
    "A4": "Detached decoder + latent drift",
    "B3": "Stop-gradient decoder control",
    "B2": "Ridge-head baseline",
}
QED_SEED_LABELS = {
    "A8": "Decoder coupling (no diversity)",
    "A6": "Decoder coupling (single $\\tau$)",
    "F": "DriftingMol",
    "G4": "Layer-balanced decoder coupling",
}
QED_PLOT_LABELS = {
    "A8": "Decoder, no z-div.",
    "A6": "Decoder, single-temp",
    "F": "DriftingMol",
    "A1": "Property head",
    "C2": "LatentMAE-guided",
    "C3": "LatentMAE + latent",
    "C1": "LatentMAE feature",
    "C5": "LatentMAE + decoder guidance",
    "A2": "Latent-space",
    "C4": "LatentMAE + decoder",
    "B1": "Random features",
    "A3": "Detached decoder",
    "A4": "Detached decoder + latent",
    "B3": "Stop-gradient decoder",
    "B2": "Ridge head",
}
QED_GROUPS = {
    "decoder": {"A8", "A6", "F"},
    "baseline": {"A1", "B2"},
    "proxy": {"C2", "C3", "C1", "C5", "A2", "C4", "B1"},
    "broken": {"A3", "A4", "B3"},
}
QED_GROUP_COLORS = {
    "decoder": "#2166AC",
    "baseline": "#6F7780",
    "proxy": "#2A8C74",
    "broken": "#B33F4A",
}

MULTI4_ORDER = ["A6", "A8", "F", "G4", "A2", "B1", "NoDrift", "B3"]
MULTI4_LABELS = {
    "A6": "Decoder coupling (single $\\tau$)",
    "A8": "Decoder coupling (no diversity)",
    "F": "DriftingMol",
    "G4": "Layer-balanced decoder coupling",
    "A2": "Latent-space drift",
    "B1": "Random-feature drift control",
    "NoDrift": "No-drift baseline",
    "B3": "Stop-gradient decoder control",
}
PROPS = [
    ("qed_rho", "QED"),
    ("sa_score_rho", "SA"),
    ("logp_rho", "LogP"),
    ("molwt_rho", "MolWt"),
]
QED_SEED_VARIANTS = ["A8", "A6", "F", "G4"]
QED_SEED_SHORT_LABELS = {
    "A6": "Single-temp",
    "A8": "No z-div.",
    "F": "DriftingMol",
    "G4": "Balanced layers",
}
ZDIV_RE = re.compile(r"zdiv(?P<value>[0-9]+p[0-9]+)")


def _float(row: dict[str, str], key: str) -> float:
    val = row.get(key, "")
    return float(val) if val not in ("", "-", "nan") else float("nan")


def read_rows() -> list[dict[str, str]]:
    with RESULTS_CSV.open() as f:
        return list(csv.DictReader(f))


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


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_atomic(fig, FIG_DIR / f"{stem}.pdf")
    savefig_atomic(fig, FIG_DIR / f"{stem}.png", dpi=300)
    print(f"Saved: {FIG_DIR / f'{stem}.pdf'}")
    print(f"Saved: {FIG_DIR / f'{stem}.png'}")


def remove_stale(stem: str) -> None:
    for suffix in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{suffix}"
        if path.exists():
            path.unlink()
            print(f"Removed stale: {path}")


def main_qed_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        r["variant"]: r
        for r in rows
        if r["condition"] == "qed"
        and r["status"].startswith("complete")
        and r["variant"] in QED_ORDER
        and not r.get("manifest_group")
        and r.get("root", "final") in {"final", "final_phi"}
    }


def plot_qed_ablation(rows: list[dict[str, str]]) -> None:
    by_variant = main_qed_rows(rows)
    variants = [v for v in QED_ORDER if v in by_variant]
    rhos = np.array([_float(by_variant[v], "spearman_rho") for v in variants])
    labels = [QED_PLOT_LABELS[v] for v in variants]

    def group_for(variant: str) -> str:
        for group, members in QED_GROUPS.items():
            if variant in members:
                return group
        return "proxy"

    colors = [QED_GROUP_COLORS[group_for(v)] for v in variants]

    fig, ax = plt.subplots(figsize=(7.05, 3.35))
    y = np.arange(len(variants))
    band_specs = [
        (-0.5, 2.5, QED_GROUP_COLORS["decoder"], "decoder-coupled"),
        (2.5, 3.5, QED_GROUP_COLORS["baseline"], "property-head"),
        (3.5, 10.5, QED_GROUP_COLORS["proxy"], "proxy / latent"),
        (10.5, len(variants) - 0.5, QED_GROUP_COLORS["broken"], "broken gradient"),
    ]
    for lo, hi, color, _label in band_specs:
        ax.axhspan(lo, hi, color=color, alpha=0.050, zorder=0)
    for boundary in [2.5, 3.5, 10.5]:
        ax.axhline(boundary, color="#D7DDE3", linewidth=0.75, zorder=1)
    for yi, rho, color in zip(y, rhos, colors):
        ax.hlines(yi, 0.0, rho, color=color, linewidth=2.2, alpha=0.76, zorder=2)
    ax.scatter(
        rhos,
        y,
        s=34,
        c=colors,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_xlabel("QED Spearman $\\rho$")
    ax.set_xlim(0, 0.56)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=7.2, pad=2)
    ax.tick_params(axis="x", labelsize=7.6)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E7EBEF", linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB6BF")
    ax.spines["bottom"].set_color("#AEB6BF")

    for yi, rho, color in zip(y, rhos, colors):
        ax.text(
            min(rho + 0.012, 0.548),
            yi,
            f"{rho:.3f}",
            ha="left",
            va="center",
            fontsize=6.5,
            color="#25313B" if rho < 0.49 else color,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.245, right=0.985, top=0.970, bottom=0.140)
    save(fig, "fig2_qed_ablation")
    plt.close(fig)


def plot_multi4_heatmap(rows: list[dict[str, str]]) -> None:
    by_variant = {}
    for row in rows:
        if row["condition"] != "multi4" or not row["status"].startswith("complete"):
            continue
        if row["root"] == "final_v2" or (row["root"] == "multi4" and row["variant"] == "G4"):
            by_variant[row["variant"]] = row
    variants = [v for v in MULTI4_ORDER if v in by_variant]
    data = np.array([[_float(by_variant[v], key) for key, _label in PROPS] for v in variants])
    min_u = np.array([100.0 * _float(by_variant[v], "min_uniqueness") for v in variants])

    fig = plt.figure(figsize=(7.05, 2.65))
    ax = fig.add_axes([0.205, 0.18, 0.455, 0.73])
    im = ax.imshow(data, cmap="YlGnBu", vmin=-0.05, vmax=0.80, aspect="auto")
    ax.set_xticks(np.arange(len(PROPS)))
    ax.set_xticklabels([label for _key, label in PROPS])
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([MULTI4_LABELS[v] for v in variants])
    ax.tick_params(axis="x", labelsize=7.4)
    ax.tick_params(axis="y", labelsize=7.3, pad=3)
    ax.set_xticks(np.arange(-0.5, len(PROPS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt_color = "white" if data[i, j] >= 0.58 else "#111111"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=6.6, color=txt_color,
                    fontweight="bold" if data[i, j] >= 0.58 else "normal")

    cax = fig.add_axes([0.682, 0.18, 0.020, 0.73])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\\rho$", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6.4, length=2)
    cb.outline.set_edgecolor("#AEB6BF")

    ax_u = fig.add_axes([0.765, 0.18, 0.185, 0.73], sharey=ax)
    y = np.arange(len(variants))
    ax_u.barh(y, min_u, color="#5E8C61", height=0.58)
    ax_u.set_xlim(0, 105)
    ax_u.set_xticks([0, 50, 100])
    ax_u.set_xlabel("Lowest U (%)")
    ax_u.tick_params(axis="y", left=False, labelleft=False)
    ax_u.tick_params(axis="x", labelsize=6.4, length=2)
    ax_u.grid(axis="x", color="#E7EBEF", linewidth=0.65)
    ax_u.set_axisbelow(True)
    ax_u.spines["top"].set_visible(False)
    ax_u.spines["right"].set_visible(False)
    ax_u.spines["left"].set_visible(False)
    ax_u.spines["bottom"].set_color("#AEB6BF")
    for yi, val in zip(y, min_u):
        ax_u.text(min(val + 2, 101), yi, f"{val:.1f}", va="center", fontsize=6.2)

    save(fig, "fig3_multi4_v2")
    plt.close(fig)


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def ci95(std: float, n: int) -> float:
    if n < 2:
        return 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    return tcrit * std / math.sqrt(n)


def plot_qed_seed_ci(rows: list[dict[str, str]]) -> None:
    by_variant_seed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["condition"] != "qed" or not row["status"].startswith("complete"):
            continue
        if row["variant"] not in QED_SEED_VARIANTS or not row.get("seed"):
            continue
        canonical = row["root"] == "final" or row.get("manifest_group") == "qed_3seed"
        if not canonical:
            continue
        key = (row["variant"], row["seed"])
        previous = by_variant_seed.get(key)
        if previous is None or row["root"] == "seeds":
            by_variant_seed[key] = row

    values_by_variant: dict[str, list[float]] = {}
    for variant in QED_SEED_VARIANTS:
        values_by_variant[variant] = [
            _float(row, "spearman_rho")
            for (v, _seed), row in by_variant_seed.items()
            if v == variant and not math.isnan(_float(row, "spearman_rho"))
        ]

    missing = {
        variant: len(values)
        for variant, values in values_by_variant.items()
        if len(values) < 3
    }
    if missing:
        have = ", ".join(f"{variant}={n}/3" for variant, n in missing.items())
        remove_stale("fig4_qed_seed_ci")
        print(f"Skipped fig4_qed_seed_ci: need 3 completed seeds per key variant; have {have}.")
        return

    summaries = []
    for variant, values in values_by_variant.items():
        mean, std = mean_std(values)
        summaries.append((variant, len(values), mean, std, ci95(std, len(values)), sorted(values)))

    summaries.sort(key=lambda item: item[2], reverse=True)
    labels = [
        QED_SEED_SHORT_LABELS.get(v, QED_SEED_LABELS.get(v, QED_LABELS.get(v, v)))
        for v, _n, _m, _s, _ci, _vals in summaries
    ]
    means = np.array([m for _v, _n, m, _s, _ci, _vals in summaries])
    cis = np.array([ci for _v, _n, _m, _s, ci, _vals in summaries])

    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    y = np.arange(len(summaries))
    colors = ["#2166AC" if variant != "G4" else "#6F7780" for variant, *_ in summaries]
    for i in range(len(summaries)):
        if i % 2 == 0:
            ax.axhspan(i - 0.46, i + 0.46, color="#F6F8FA", zorder=0)

    for i, (summary, mean, ci, color) in enumerate(zip(summaries, means, cis, colors)):
        _variant, _n, _m, _std, _ci, seed_values = summary
        jitter = np.linspace(-0.13, 0.13, len(seed_values))
        ax.scatter(
            seed_values,
            i + jitter,
            s=13,
            color=color,
            alpha=0.35,
            linewidth=0,
            zorder=2,
        )
        ax.errorbar(
            mean,
            i,
            xerr=ci,
            fmt="o",
            color=color,
            ecolor="#26333F",
            elinewidth=0.95,
            capsize=2.8,
            capthick=0.95,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("QED Spearman $\\rho$")
    ax.set_ylim(len(summaries) - 0.55, -0.45)
    ax.set_xlim(
        max(0.40, float(np.nanmin(means - cis)) - 0.018),
        min(0.60, float(np.nanmax(means + cis)) + 0.035),
    )
    ax.tick_params(axis="y", labelsize=7.2, pad=2)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E3E8EE", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB6BF")
    ax.spines["bottom"].set_color("#AEB6BF")
    for i, (mean, ci, color) in enumerate(zip(means, cis, colors)):
        ax.text(
            min(mean + ci + 0.012, ax.get_xlim()[1] - 0.002),
            i,
            f"{mean:.3f}",
            ha="left",
            va="center",
            fontsize=6.3,
            color=color,
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.315, right=0.965, top=0.965, bottom=0.255)
    save(fig, "fig4_qed_seed_ci")
    plt.close(fig)


def plot_zdiv_pareto(rows: list[dict[str, str]]) -> None:
    points = []
    for row in rows:
        if row["root"] != "zdiv" or not row["status"].startswith("complete"):
            continue
        match = ZDIV_RE.search(row["experiment"])
        if not match:
            continue
        zdiv = float(match.group("value").replace("p", "."))
        points.append((zdiv, _float(row, "spearman_rho"), 100.0 * _float(row, "uniqueness")))

    if len(points) < 2:
        remove_stale("fig5_zdiv_pareto")
        print("Skipped fig5_zdiv_pareto: fewer than two completed z-div points.")
        return

    points.sort()
    zdiv = np.array([p[0] for p in points])
    rho = np.array([p[1] for p in points])
    uniq = np.array([p[2] for p in points])

    x = np.arange(len(points))
    fig, (ax, ax_u) = plt.subplots(
        2,
        1,
        figsize=(3.45, 2.35),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12},
    )
    ax.plot(
        x,
        rho,
        color="#2166AC",
        marker="o",
        markersize=4.8,
        linewidth=1.45,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )
    ax_u.plot(
        x,
        uniq,
        color="#C45A2D",
        marker="s",
        markersize=4.4,
        linewidth=1.35,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )
    ax_u.axhspan(98.0, 98.35, color="#C45A2D", alpha=0.065, zorder=0)
    ax_u.set_xlabel("z-diversity weight $\\lambda_z$")
    ax.set_ylabel("$\\rho$", color="#2166AC")
    ax_u.set_ylabel("U (%)", color="#C45A2D")
    ax.set_xticks(x)
    ax_u.set_xticks(x)
    ax_u.set_xticklabels([f"{val:g}" for val in zdiv])
    ax.set_ylim(float(np.nanmin(rho)) - 0.005, float(np.nanmax(rho)) + 0.007)
    ax_u.set_ylim(float(np.nanmin(uniq)) - 0.45, float(np.nanmax(uniq)) + 0.45)
    for panel_ax in (ax, ax_u):
        panel_ax.set_axisbelow(True)
        panel_ax.grid(axis="y", color="#E3E8EE", linewidth=0.65)
        panel_ax.spines["top"].set_visible(False)
        panel_ax.spines["right"].set_visible(False)
        panel_ax.spines["left"].set_color("#AEB6BF")
        panel_ax.spines["bottom"].set_color("#AEB6BF")
        panel_ax.tick_params(axis="both", labelsize=7.0)
    ax.tick_params(axis="x", labelbottom=False)
    ax_u.tick_params(axis="y", labelsize=7.0, colors="#C45A2D")
    ax.tick_params(axis="y", colors="#2166AC")
    fig.subplots_adjust(left=0.165, right=0.965, top=0.975, bottom=0.235, hspace=0.12)
    save(fig, "fig5_zdiv_pareto")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    plot_qed_ablation(rows)
    plot_multi4_heatmap(rows)
    plot_qed_seed_ci(rows)
    plot_zdiv_pareto(rows)


if __name__ == "__main__":
    main()
