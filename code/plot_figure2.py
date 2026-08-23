#!/usr/bin/env python3
"""Reproduce all three panels used in Fig. 2 from the benchmark CSV files.

Run the three benchmark programs first, then
    python plot_figure2.py --data-dir results --figure-dir figures

Inputs
    electronic_convergence.csv
    photon_convergence.csv
    joint_primary.csv

Outputs
    electron_exact_limit.pdf/.png
    photon_exact_limit.pdf/.png
    joint_analytic_inverse.pdf/.png

Dependencies: numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_electronic(rows, out: Path):
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    specs = [("balanced", "o", "-"), ("asymmetric", "s", "--")]
    for sequence, marker, linestyle in specs:
        data = sorted((r for r in rows if r["sequence"] == sequence), key=lambda r: int(r["distance"]))
        ax.semilogy(
            [int(r["distance"]) for r in data],
            [float(r["source_error"]) for r in data],
            marker=marker,
            linestyle=linestyle,
            label=sequence,
        )
    ax.set_xlim(2.7, 9.35)
    ax.set_xlabel(r"nearest-boundary parameter $d$")
    ax.set_ylabel(r"$\max_t|v_d(t)-v_\infty(t)|$")
    ax.legend(loc="upper right", frameon=False, fontsize=7.2, handlelength=2.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(labelsize=7.5)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    fig.savefig(out / "electron_exact_limit.pdf", bbox_inches="tight")
    fig.savefig(out / "electron_exact_limit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_photon(rows, out: Path):
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    specs = [
        ("gauss-legendre", "o", "-", "Gauss--Legendre"),
        ("midpoint", "s", "--", "positive midpoint"),
    ]
    for family, marker, linestyle, label in specs:
        data = sorted((r for r in rows if r["family"] == family), key=lambda r: int(r["order"]))
        ax.loglog(
            [int(r["order"]) for r in data],
            [float(r["source_error"]) for r in data],
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xlim(1.8, 13.2)
    plotted_orders = [2, 3, 4, 6, 8, 12]
    ax.set_xticks(plotted_orders)
    ax.set_xticklabels([str(v) for v in plotted_orders])
    ax.minorticks_off()
    ax.set_xlabel(r"photon quadrature order $M$")
    ax.set_ylabel(r"$\max_t|j_M(t)-j_\infty(t)|$")
    ax.legend(loc="upper right", frameon=False, fontsize=7.0, handlelength=2.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(labelsize=7.5)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    fig.savefig(out / "photon_exact_limit.pdf", bbox_inches="tight")
    fig.savefig(out / "photon_exact_limit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_joint(rows, out: Path):
    L_values = [4, 8, 12]
    M_values = [1, 2, 4]
    lookup = {(int(r["L"]), int(r["M"])): float(r["E_src"]) for r in rows}
    values = np.array([[lookup[(L, M)] for L in L_values] for M in M_values], dtype=float)

    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    image = ax.imshow(np.log10(values), origin="lower", aspect="auto")
    ax.set_xticks(range(len(L_values)), [str(v) for v in L_values])
    ax.set_yticks(range(len(M_values)), [str(v) for v in M_values])
    ax.set_xlabel(r"electronic regulator $L$")
    ax.set_ylabel(r"photon regulator $M$")
    threshold = float(np.mean(np.log10(values)))
    for i in range(len(M_values)):
        for j in range(len(L_values)):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:.1e}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if np.log10(value) < threshold else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"$\log_{10}E^{\rm src}_{L,M}$", fontsize=8.2)
    colorbar.ax.tick_params(labelsize=7.0)
    ax.tick_params(labelsize=7.5)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    fig.savefig(out / "joint_analytic_inverse.pdf", bbox_inches="tight")
    fig.savefig(out / "joint_analytic_inverse.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    plot_electronic(read_csv(args.data_dir / "electronic_convergence.csv"), args.figure_dir)
    plot_photon(read_csv(args.data_dir / "photon_convergence.csv"), args.figure_dir)
    plot_joint(read_csv(args.data_dir / "joint_primary.csv"), args.figure_dir)
    print(f"wrote Fig. 2 panel files to {args.figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
