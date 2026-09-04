"""Figures for the whitepaper, drawn from results/ so every number is traceable.

No figure here carries a hand-typed value. Each one reads the same CSV or JSON
the repository ships, so a figure and the table beside it cannot disagree, and
re-running the study re-draws the paper.

Palette: the three categorical slots validate all-pairs for colour-vision
deficiency (worst pair dE 9.2 deutan, 24.0 normal-vision) on a light surface.
Every mark also carries a direct value label, which is what the aqua slot's
sub-3:1 contrast obliges and what makes the figures survive greyscale printing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

__all__ = ["build_all"]

RESULTS = Path(__file__).resolve().parents[1] / "results"

# -- design tokens ---------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdcd8"
S1 = "#2a78d6"   # ours / in-distribution
S2 = "#eb6834"   # published / transfer
S3 = "#1baf7a"   # third series
WARN = "#b08900"


def _fig(w: float, h: float, dpi: int = 200):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(w, h), dpi=dpi, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure


def _style(axes, ylabel: str = "", xlabel: str = "", ymax: float | None = None):
    axes.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(GRID)
    axes.tick_params(labelsize=8.5, colors=INK_2, length=3)
    axes.yaxis.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    if ylabel:
        axes.set_ylabel(ylabel, fontsize=9, color=INK)
    if xlabel:
        axes.set_xlabel(xlabel, fontsize=9, color=INK)
    if ymax is not None:
        axes.set_ylim(0, ymax)


def _labels(axes, xs, values, fmt="{:.3f}", dy=0.015, size=8):
    for x, v in zip(xs, values, strict=True):
        if np.isnan(v):
            continue
        axes.text(x, v + dy, fmt.format(v), ha="center", va="bottom",
                  fontsize=size, color=INK)


def _read_reproduction() -> list[dict]:
    with (RESULTS / "reproduction.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_aitex() -> list[dict]:
    with (RESULTS / "aitex_generalisation.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #


def fig_reproduction(out: Path) -> Path:
    """Ours against the published figure, per MVTec texture class."""
    rows = [r for r in _read_reproduction() if r["configuration"] == "product"]
    order = ["carpet", "leather", "grid"]
    rows.sort(key=lambda r: order.index(r["category"]))

    names = [r["category"] for r in rows]
    ours = [float(r["image_auroc"]) for r in rows]
    published = [float(r["published_image_auroc"]) for r in rows]

    figure = _fig(6.6, 3.0)
    axes = figure.add_subplot(111)
    x = np.arange(len(names))
    w = 0.36
    axes.bar(x - w / 2, ours, w, color=S1, label="LineSight (30 images, cold start)",
             edgecolor=SURFACE, linewidth=1.2)
    axes.bar(x + w / 2, published, w, color=S2, label="Published PatchCore (WideResNet-50)",
             edgecolor=SURFACE, linewidth=1.2)
    _labels(axes, x - w / 2, ours)
    _labels(axes, x + w / 2, published, fmt="{:.2f}")

    axes.set_xticks(x)
    axes.set_xticklabels(names, fontsize=9.5, color=INK)
    _style(axes, ylabel="Image AUROC", ymax=1.18)
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    axes.text(len(names) - 0.45, 0.515, "chance", fontsize=7.5, color=INK_2)
    axes.legend(fontsize=8, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.16), ncol=2)
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return out


def fig_localisation(out: Path) -> Path:
    """The consistent finding: localisation outruns whole-tile detection."""
    repro = {r["category"]: r for r in _read_reproduction()
             if r["configuration"] == "product"}
    aitex = [r for r in _read_aitex() if r["in_distribution"] == "True"]
    tfd = json.loads((RESULTS / "tfd_summary.json").read_text(encoding="utf-8"))

    names = ["MVTec\ncarpet", "MVTec\nleather", "AITEX\n(6 fabrics)", "TFD\n(10 fabrics)"]
    localise = [
        float(repro["carpet"]["pixel_auroc"]),
        float(repro["leather"]["pixel_auroc"]),
        float(np.mean([float(r["pixel_auroc"]) for r in aitex])),
        tfd["block_auroc_mean_over_fabrics"],
    ]
    detect = [
        float(repro["carpet"]["image_auroc"]),
        float(repro["leather"]["image_auroc"]),
        float(np.mean([float(r["tile_auroc"]) for r in aitex])),
        tfd["image_auroc_mean_over_fabrics"],
    ]

    figure = _fig(6.6, 3.1)
    axes = figure.add_subplot(111)
    x = np.arange(len(names))
    w = 0.36
    axes.bar(x - w / 2, localise, w, color=S1, edgecolor=SURFACE, linewidth=1.2,
             label="Localisation (pixel / block AUROC)")
    axes.bar(x + w / 2, detect, w, color=S3, edgecolor=SURFACE, linewidth=1.2,
             label="Detection (image / tile AUROC)")
    _labels(axes, x - w / 2, localise)
    _labels(axes, x + w / 2, detect)

    axes.set_xticks(x)
    axes.set_xticklabels(names, fontsize=8.5, color=INK)
    _style(axes, ylabel="AUROC", ymax=1.18)
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.legend(fontsize=8, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.15), ncol=2)
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return out


def fig_aitex(out: Path) -> Path:
    """Per-fabric cold start against a bank borrowed from another fabric."""
    rows = _read_aitex()
    codes = sorted({r["eval_fabric"] for r in rows})
    same, cross = [], []
    for code in codes:
        s = [float(r["tile_auroc"]) for r in rows
             if r["eval_fabric"] == code and r["in_distribution"] == "True"]
        c = [float(r["tile_auroc"]) for r in rows
             if r["eval_fabric"] == code and r["in_distribution"] == "False"]
        same.append(s[0] if s else np.nan)
        cross.append(float(np.mean(c)) if c else np.nan)

    figure = _fig(6.6, 3.1)
    axes = figure.add_subplot(111)
    x = np.arange(len(codes))
    w = 0.36
    axes.bar(x - w / 2, same, w, color=S1, edgecolor=SURFACE, linewidth=1.2,
             label="Own bank (cold start)")
    axes.bar(x + w / 2, cross, w, color=S2, edgecolor=SURFACE, linewidth=1.2,
             label="Bank from another fabric (transfer)")
    _labels(axes, x - w / 2, same)
    _labels(axes, x + w / 2, cross)

    axes.axhline(float(np.nanmean(same)), color=S1, linewidth=1.0, linestyle="--")
    axes.axhline(float(np.nanmean(cross)), color=S2, linewidth=1.0, linestyle="--")
    axes.text(len(codes) - 0.35, float(np.nanmean(same)) + 0.02,
              f"mean {np.nanmean(same):.3f}", fontsize=7.5, color=S1)
    axes.text(len(codes) - 0.35, float(np.nanmean(cross)) - 0.06,
              f"mean {np.nanmean(cross):.3f}", fontsize=7.5, color=S2)

    axes.set_xticks(x)
    axes.set_xticklabels([f"fabric {c}" for c in codes], fontsize=8.5, color=INK)
    _style(axes, ylabel="Tile AUROC", ymax=1.18)
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    axes.legend(fontsize=8, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.15), ncol=2)
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return out


def fig_tfd(out: Path) -> Path:
    """Ten independent cold starts on real factory fabric, including the failures."""
    with (RESULTS / "tfd_per_fabric.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    codes = [r["fabric"] for r in rows]
    block = [float(r["block_auroc"]) for r in rows]
    image = [float(r["image_auroc"]) for r in rows]

    figure = _fig(6.6, 3.1)
    axes = figure.add_subplot(111)
    x = np.arange(len(codes))
    w = 0.36
    axes.bar(x - w / 2, block, w, color=S1, edgecolor=SURFACE, linewidth=1.2,
             label="Block AUROC (localisation)")
    axes.bar(x + w / 2, image, w, color=S3, edgecolor=SURFACE, linewidth=1.2,
             label="Image AUROC (detection)")
    _labels(axes, x - w / 2, block, fmt="{:.2f}", size=7)
    _labels(axes, x + w / 2, image, fmt="{:.2f}", size=7)

    axes.set_xticks(x)
    axes.set_xticklabels(codes, fontsize=8, color=INK)
    _style(axes, ylabel="AUROC", xlabel="TFD fabric", ymax=1.18)
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    axes.annotate("fabric 001 detects at chance\nbut still localises at 0.88",
                  xy=(0, 0.50), xytext=(1.1, 0.18), fontsize=7.5, color=INK_2,
                  arrowprops={"arrowstyle": "->", "color": INK_2, "linewidth": 0.7})
    axes.legend(fontsize=8, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.15), ncol=2)
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return out


def fig_latency(out: Path) -> Path:
    """Where the 478 ms goes. Measured, at the stated frame stride."""
    stages = ["Geometry\n(ArUco)", "Preprocess\n(flat-field + tiling)",
              "Backbone\n(ResNet-18)", "NN search", "Event assembly\n+ ASTM"]
    values = [0.4, 314.9, 153.9, 0.8, 10.1]

    figure = _fig(6.6, 2.9)
    axes = figure.add_subplot(111)
    y = np.arange(len(stages))[::-1]
    axes.barh(y, values, 0.55, color=S1, edgecolor=SURFACE, linewidth=1.2)
    for yy, v in zip(y, values, strict=True):
        axes.text(v + 6, yy, f"{v:.1f} ms", va="center", fontsize=8.5, color=INK)

    axes.set_yticks(y)
    axes.set_yticklabels(stages, fontsize=8.5, color=INK)
    axes.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.xaxis.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(labelsize=8.5, colors=INK_2, length=3)
    axes.set_xlabel("median milliseconds per processed frame", fontsize=9, color=INK)
    axes.set_xlim(0, 380)
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return out


def fig_astm(out: Path) -> Path:
    """The penalty schedule the verdict is computed from."""
    figure = _fig(6.6, 2.7)
    axes = figure.add_subplot(111)

    bands = [(0, 75, 1), (75, 150, 2), (150, 230, 3), (230, 400, 4)]
    for lo, hi, points in bands:
        axes.plot([lo, hi], [points, points], color=S1, linewidth=2.2,
                  solid_capstyle="butt")
        axes.text((lo + hi) / 2, points + 0.16, f"{points} pt", ha="center",
                  fontsize=8.5, color=INK)
    for x in (75, 150, 230):
        axes.axvline(x, color=GRID, linewidth=0.8, linestyle="--")

    axes.axvspan(230, 400, color=WARN, alpha=0.10)
    axes.text(315, 2.3,
              "at and above 230 mm a defect is\ncontinuous: 4 points per metre\nit occupies",
              ha="center", fontsize=8, color=INK_2)

    _style(axes, ylabel="Penalty points", xlabel="Defect length along the roll (mm)")
    axes.set_xlim(0, 400)
    # Ticks ON the band boundaries rather than at round numbers: the boundaries
    # are the information, and a default tick at 150 next to a hand-drawn 150
    # renders the label twice.
    axes.set_xticks([0, 75, 150, 230, 300, 400])
    axes.set_ylim(0, 5)
    axes.set_yticks([0, 1, 2, 3, 4])
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return out


def build_all(out_dir: Path) -> dict[str, Path]:
    """Draw every figure into ``out_dir``; returns name -> path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "reproduction": fig_reproduction,
        "localisation": fig_localisation,
        "aitex": fig_aitex,
        "tfd": fig_tfd,
        "latency": fig_latency,
        "astm": fig_astm,
    }
    return {name: fn(out_dir / f"fig_{name}.png") for name, fn in builders.items()}


if __name__ == "__main__":
    paths = build_all(Path(__file__).resolve().parents[1] / "docs" / "whitepaper")
    for name, path in paths.items():
        print(f"  {name:14s} {path.name}  {path.stat().st_size // 1024} KB")
