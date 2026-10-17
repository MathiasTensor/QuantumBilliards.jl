#!/usr/bin/env python3
"""Plot exact circular-sector convergence without hiding missed eigenvalues.

Usage:
  python benchmarks/plot_weighted_corner_angles.py RESULTS/convergence.csv
  python benchmarks/plot_weighted_corner_angles.py RESULTS/convergence.csv \
      --runtime RESULTS/optimization_runtime.csv

The Julia driver writes raw measurements. This script only displays them; it
does not replace imaginary parts by zero, discard failed counts, smooth curves,
or claim that a tiny imaginary part alone certifies eigenvalue accuracy.
Final figures contain their own explanations. The figures directory contains
PDFs only; PNG previews go to a temporary directory or the --previews path.
"""

import argparse
import csv
from pathlib import Path
import re
import tempfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import numpy as np


KINDS = {
    "weighted": ("Weighted physical density", "-", "o", "black"),
    "legacy_q4": ("Previous graded Kress (q=4)", "--", "s", "0.48"),
}
PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None,
                "ModDate": None, "Title": None, "Author": None,
                "Subject": None, "Keywords": None}


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def complete(row):
    return (row.get("status") in ("complete", "ok")
            and number(row, "found_count") == number(row, "expected_count")
            and number(row, "expected_count") > 0)


def base_rows(rows):
    return [row for row in rows if number(row, "source_refinement", 1) == 1]


def angle_key(row):
    return int(row["angle_numerator"]), int(row["angle_denominator"])


def angle_label(key):
    numerator, denominator = key
    numerator = "" if numerator == 1 else str(numerator)
    return rf"$\alpha={numerator}\pi/{denominator}$"


def sector_inset(ax, angle):
    """Show the actual sector; its two other corners are right angles."""
    # Geometry belongs in the heading, where it cannot hide an error curve.
    inset = ax.inset_axes([0.78, 1.02, 0.22, 0.22])
    theta = np.linspace(0, angle, 200)
    points = np.column_stack((np.cos(theta), np.sin(theta)))
    inset.plot(np.r_[0, points[:, 0], 0], np.r_[0, points[:, 1], 0],
               color="0.45", lw=0.65)
    inset.add_patch(Arc((0, 0), 0.42, 0.42, theta1=0,
                        theta2=np.degrees(angle), lw=0.7, color="black"))
    inset.plot([0], [0], "o", color="black", ms=1.8)
    inset.set_aspect("equal")
    inset.set_xlim(min(-0.12, points[:, 0].min() - 0.08), 1.1)
    inset.set_ylim(min(-0.12, points[:, 1].min() - 0.08),
                   max(0.12, points[:, 1].max() + 0.08))
    inset.axis("off")


def caption_text(rows, metric):
    valid = [row for row in rows if complete(row)]
    failed = [row for row in rows if not complete(row)]
    failures = (f"{len(failed)} solves have incomplete or failed root counts; "
                "crosses identify these and lines are broken there. "
                if failed else "Every plotted solve retains the full exact root count. ")
    if metric == "max_imaginary_part":
        meaning = ("A real Dirichlet spectrum should have zero imaginary part. "
                   "This diagnostic is necessary but cannot detect a wrong real part; "
                   "compare the exact-error figure. ")
    elif metric == "nodes":
        meaning = ("Actual boundary nodes determine the dense matrix dimension. "
                   "Leftward curves achieve an error with fewer unknowns; nominal "
                   "ppw can hide the corner-panel safety floor. ")
    else:
        meaning = ("Error is max |k_computed - k_exact| over every root in the "
                   "contour, retaining complex parts and multiplicities. "
                   "Exact roots are zeros of J_(m*pi/alpha)(k), m >= 1. ")
    better = []
    for key in sorted({angle_key(row) for row in valid}, key=lambda a: a[0]/a[1]):
        minimum = {}
        for kind in KINDS:
            good = [number(row, "nodes") for row in valid
                    if angle_key(row) == key and row["kind"] == kind
                    and number(row, "max_root_error") <= 1e-8]
            if good:
                minimum[kind] = min(good)
        if len(minimum) == 2:
            better.append(1 - minimum["weighted"] / minimum["legacy_q4"])
    if better and metric != "max_imaginary_part":
        savings = (f"At the 1e-8 target, the measured weighted-node saving spans "
                   f"{100*min(better):.0f}% to {100*max(better):.0f}% among "
                   f"{len(better)} angles where both methods reach the target. ")
    else:
        savings = ""
    return meaning + failures + savings + (
        "Radius-1 Dirichlet sectors have two additional pi/2 joins. Insets mark the varied corner. "
        "Weighting factors out the leading q(r) ~ r^(lambda-1) power, lambda = pi/alpha; "
        "higher powers and possible logarithms remain, so exponential convergence is not assumed.")


def save_figure(fig, output, preview):
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata=PDF_METADATA)
    fig.savefig(preview, dpi=190, facecolor="white")
    plt.close(fig)
    print(output)


def convergence_figure(rows, metric, output, preview):
    groups = sorted({angle_key(row) for row in rows}, key=lambda a: a[0]/a[1])
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 5.5), squeeze=False)
    is_nodes = metric == "nodes"
    ykey = "max_root_error" if is_nodes else metric
    xkey = "nodes" if is_nodes else "requested_ppw"
    finite = [number(row, ykey) for row in rows if complete(row)
              and np.isfinite(number(row, ykey))]
    floor = 1e-15
    ymin = min(floor, max(min(finite, default=floor) / 4, 1e-17))
    ymax = max(1e-6, max(finite, default=1e-6) * 4)
    for ax, key in zip(axes.flat, groups):
        subset = [row for row in rows if angle_key(row) == key]
        for kind, (_, line, marker, color) in KINDS.items():
            values = sorted([row for row in subset if row["kind"] == kind],
                            key=lambda row: number(row, xkey))
            xs = [number(row, xkey) for row in values]
            ys = [max(number(row, ykey), floor) if complete(row)
                  and np.isfinite(number(row, ykey)) else np.nan for row in values]
            ax.plot(xs, ys, color=color, ls=line, marker=marker, ms=3,
                    lw=0.85, markerfacecolor="white", markeredgewidth=0.7)
            failed = [row for row in values if not complete(row)]
            if failed:
                ax.plot([number(row, xkey) for row in failed],
                        [ymax / 1.7] * len(failed), ls="none", marker="x",
                        color=color, ms=4, mew=0.9)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        if ykey == "max_root_error":
            ax.axhline(1e-8, color="0.72", lw=0.65, ls=":", zorder=0)
        exponent = str(key[1]) if key[0] == 1 else f"{key[1]}/{key[0]}"
        ax.set_title(angle_label(key) + rf", $\lambda={exponent}$",
                     fontsize=7.7, pad=2, loc="left", y=1.02)
        ax.grid(axis="y", which="major", color="0.9", lw=0.45)
        ax.tick_params(labelsize=6.5, length=2.5, width=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        sector_inset(ax, np.pi * key[0] / key[1])
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    short_labels = ("Weighted density\n(physical normal q)", "Previous graded Kress\n(grading power 4)")
    handles = [Line2D([], [], color=color, ls=line, marker=marker, ms=3,
                      lw=0.9, markerfacecolor="white", label=label)
               for label, (_, line, marker, color) in zip(short_labels, KINDS.values())]
    handles.append(Line2D([], [], color="black", marker="x", ls="none", ms=4,
                          label="Incomplete / failed\nroot count"))
    legend_ax = list(axes.flat)[-1]
    if len(groups) < len(axes.flat):
        legend_ax.legend(handles=handles, loc="center", frameon=False,
                         fontsize=6.4, handlelength=2.1, labelspacing=1.2)
        legend_ax.text(0.06, 0.12,
                       "Display floor: 1e-15.\nRoots keep their\ncomplex parts.",
                       transform=legend_ax.transAxes, fontsize=6.1, va="top")
    else:
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.5, 0.93))
    title = {"max_root_error": "Corner weighting: convergence against exact eigenvalues",
             "max_imaginary_part": "Corner weighting: imaginary parts of the computed roots",
             "nodes": "Corner weighting: accuracy per actual boundary unknown"}[metric]
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.supxlabel("Actual boundary nodes N" if is_nodes else "Requested points per wavelength (ppw)",
                   fontsize=8, y=0.205)
    fig.supylabel(r"Maximum $|\mathrm{Im}\,k|$" if ykey == "max_imaginary_part"
                  else r"Maximum $|k-k_{\rm exact}|$", fontsize=8, x=0.018)
    description = caption_text(rows, metric)
    fig.text(0.065, 0.135, "\n".join(textwrap.wrap(description, 139)),
             ha="left", va="top", fontsize=6.6, linespacing=1.3)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.29, top=0.905,
                        wspace=0.35, hspace=0.47)
    save_figure(fig, output, preview)


def runtime_figure(rows, output, preview, context=""):
    """Runtime input is paired medians, never inferred from convergence runs."""
    rows = [row for row in rows if number(row, "before_seconds") > 0
            and number(row, "after_seconds") > 0]
    if not rows:
        return
    labels = [row["case"].replace(", both parities", " (both)")
              .replace(", both sectors", " (both)").replace(", one sector", "") + ": " + row["part"]
              for row in rows]
    before = np.array([number(row, "before_seconds") for row in rows])
    after = np.array([number(row, "after_seconds") for row in rows])
    x = np.arange(len(rows))
    height = max(3.8, len(rows) * 0.32 + 1.8)
    fig, (ax, relative) = plt.subplots(1, 2, figsize=(7.5, height),
                                      sharey=True,
                                      gridspec_kw={"width_ratios": [1.5, 1]})
    ax.barh(x - 0.18, before, 0.34, color="white", edgecolor="black", lw=0.7,
            hatch="///", label="Before optimization")
    ax.barh(x + 0.18, after, 0.34, color="0.3", edgecolor="black", lw=0.7,
            label="After optimization")
    ax.set_yticks(x, labels, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("Median elapsed time (s; logarithmic axis)", fontsize=7)
    ax.invert_yaxis()
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7,
               loc="upper center", bbox_to_anchor=(0.63, 0.91), ncol=2)
    relative.barh(x, before / after, color="0.65", edgecolor="black", lw=0.6)
    relative.tick_params(axis="y", left=False, labelleft=False)
    relative.set_xlabel("Speedup (before / after)", fontsize=8)
    relative.axvline(1, color="black", lw=0.8, ls=":")
    for i, ratio in enumerate(before / after):
        relative.text(ratio, i, f" {ratio:.2f}x", va="center", fontsize=7)
    relative.set_xlim(0, max(before / after) * 1.25)
    for item in (ax, relative):
        item.spines[["top", "right"]].set_visible(False)
        item.tick_params(labelsize=6.5, length=2.5)
    fig.suptitle("Weighted density: measured optimization cost", fontsize=10, y=0.98)
    if context:
        fig.text(0.25, 0.835, context, fontsize=6.2, ha="left", va="bottom")
    note = ("Purpose: compare the same weighted density space and numerical accuracy before and after optimization. "
            "Bars show paired warmed timing medians; speedup includes only the named part. "
            "Geometry, roots, filters and quadrature orders are retained. "
            "These timings measure implementation changes, separately from the saving due to fewer boundary nodes.")
    if any("both" in row["case"] for row in rows):
        note += " Setup/matrix rows use one sector; full-solve rows sum both reflection sectors."
    if all(np.isfinite(number(row, "repetitions")) for row in rows):
        note += " Repetitions by row (top to bottom): " + ", ".join(
            str(int(number(row, "repetitions"))) for row in rows) + "."
    if any("beyn" in row["part"].lower()
           and abs(number(row, "after_seconds") / number(row, "before_seconds") - 1) < 0.02
           for row in rows):
        note += " Here the full Beyn solve is effectively unchanged, despite faster setup and assembly."
    fig.text(0.055, 0.155, "\n".join(textwrap.wrap(note, 139)), fontsize=6.6,
             va="top", linespacing=1.3)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.81, bottom=0.26, wspace=0.15)
    save_figure(fig, output, preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("convergence", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--figures", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs" / "src" / "figures")
    parser.add_argument("--previews", type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "savefig.facecolor": "white"})
    rows = base_rows(read_rows(args.convergence))
    if not rows:
        parser.error("no unrefined convergence records were found")
    previews = args.previews or Path(tempfile.gettempdir()) / "QuantumBilliards-weighted-corner-previews"
    for metric, stem in (("max_root_error", "weighted_corner_angles_exact_error"),
                         ("max_imaginary_part", "weighted_corner_angles_imaginary"),
                         ("nodes", "weighted_corner_angles_nodes")):
        convergence_figure(rows, metric, args.figures / (stem + ".pdf"),
                           previews / (stem + ".png"))
    if args.runtime:
        context = ""
        settings = args.runtime.parent / "before_full_settings.txt"
        micro = args.runtime.parent / "before_micro_medians.csv"
        if settings.exists() and micro.exists():
            settings_text = settings.read_text()
            configuration = re.search(
                r"Julia ([^;]+); Julia threads (\d+); initial BLAS threads (\d+); LAPACK macro budget (\d+)",
                settings_text)
            micro_rows = read_rows(micro)
            if configuration and micro_rows:
                version, threads, _, budget = configuration.groups()
                backend = "OpenBLAS" if "libopenblas" in settings_text else "BLAS"
                assembly = int(number(micro_rows[0], "blas_threads"))
                context = (f"Julia {version}; {threads} Julia threads; {backend}: "
                           f"{assembly} thread for assembly, {budget} for LAPACK")
        runtime_figure(read_rows(args.runtime),
                       args.figures / "weighted_corner_optimization_runtime.pdf",
                       previews / "weighted_corner_optimization_runtime.png", context)


if __name__ == "__main__":
    main()
