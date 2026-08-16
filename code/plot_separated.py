#!/usr/bin/env python3
"""Plot the manuscript's separate electronic and photon benchmark figures."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def electronic(data: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    for seq, marker, ls in [("balanced", "o", "-"), ("asymmetric", "s", "--")]:
        d = data[data.sequence == seq].sort_values("distance")
        line, = ax.semilogy(d.distance, d.source_error, marker=marker, linestyle=ls)
        y = float(d.source_error.iloc[-1])
        offset = (7, -6) if seq == "balanced" else (7, 6)
        ax.annotate(seq, xy=(float(d.distance.iloc[-1]), y), xytext=offset,
                    textcoords="offset points", fontsize=7.3, color=line.get_color(),
                    ha="left", va="center", clip_on=False)
    ax.set_xlim(data.distance.min() - .3, data.distance.max() + 2.2)
    ax.set_xlabel(r"nearest-boundary parameter $d$")
    ax.set_ylabel(r"$\max_t|v_d(t)-v_\infty(t)|$")
    ax.grid(True, which="both", alpha=.25)
    ax.tick_params(labelsize=7.5)
    ax.xaxis.label.set_size(8.5); ax.yaxis.label.set_size(8.5)
    fig.savefig(out / "electron_exact_limit.pdf", bbox_inches="tight")
    fig.savefig(out / "electron_exact_limit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def photon(data: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    specs = [("gauss-legendre", "o", "-", "Gauss--Legendre"),
             ("midpoint", "s", "--", "positive midpoint")]
    for fam, marker, ls, label in specs:
        d = data[(data.family == fam) & (data.order <= 12)].sort_values("order")
        line, = ax.loglog(d.order, d.source_error, marker=marker, linestyle=ls)
        x = float(d.order.iloc[-1]); y = float(d.source_error.iloc[-1])
        offset = (7, -3 if fam == "gauss-legendre" else 5)
        ax.annotate(label, xy=(x, y), xytext=offset, textcoords="offset points",
                    fontsize=7.2, color=line.get_color(), ha="left", va="center", clip_on=False)
    ax.set_xlim(1.75, 24)
    ax.set_xlabel(r"photon quadrature order $M$")
    ax.set_ylabel(r"$\max_t|j_M(t)-j_\infty(t)|$")
    ax.grid(True, which="both", alpha=.25)
    ax.tick_params(labelsize=7.5)
    ax.xaxis.label.set_size(8.5); ax.yaxis.label.set_size(8.5)
    fig.savefig(out / "photon_exact_limit.pdf", bbox_inches="tight")
    fig.savefig(out / "photon_exact_limit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=Path("results"))
    ap.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = ap.parse_args(); args.figure_dir.mkdir(parents=True, exist_ok=True)
    electronic(pd.read_csv(args.result_dir / "electronic_convergence.csv"), args.figure_dir)
    photon(pd.read_csv(args.result_dir / "photon_convergence.csv"), args.figure_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
