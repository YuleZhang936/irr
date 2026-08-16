"""Draw the electron-photon device schematic used by the manuscript."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch


def main() -> None:
    outdir = Path(__file__).resolve().parents[1] / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 2.65))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Electronic leads and fixed device.
    lead_fc = "#a9cee6"
    device_fc = "#e8c77b"
    photon_fc = "#c8a4df"
    edge = "#313131"

    ax.add_patch(Rectangle((0.55, 1.55), 1.75, 1.45, facecolor=lead_fc, edgecolor=edge, lw=1.4))
    ax.add_patch(Rectangle((4.0, 1.35), 2.0, 1.85, facecolor=device_fc, edgecolor=edge, lw=1.5))
    ax.add_patch(Rectangle((7.7, 1.55), 1.75, 1.45, facecolor=lead_fc, edgecolor=edge, lw=1.4))

    ax.plot([2.3, 4.0], [2.275, 2.275], color=edge, lw=2.0)
    ax.plot([6.0, 7.7], [2.275, 2.275], color=edge, lw=2.0)

    ax.text(1.425, 2.275, r"left lead $E_L$", ha="center", va="center", fontsize=9.5)
    ax.text(5.0, 2.275, r"device $C$", ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(8.575, 2.275, r"right lead $E_R$", ha="center", va="center", fontsize=9.5)

    # Selected photon coordinate and residual continuum.
    q_center = (5.0, 4.15)
    ax.add_patch(Circle(q_center, 0.43, facecolor=photon_fc, edgecolor=edge, lw=1.4))
    ax.text(*q_center, r"$Q$", ha="center", va="center", fontsize=12)
    ax.add_patch(FancyArrowPatch((5.0, 3.2), (5.0, 3.72), arrowstyle="<->", mutation_scale=12,
                                 linewidth=1.35, color=edge))
    ax.text(5.18, 3.48, r"$\lambda$", ha="left", va="center", fontsize=9.5)

    xs = [6.15, 6.78, 7.41, 8.04]
    radii = [0.20, 0.17, 0.145, 0.12]
    for x, r in zip(xs, radii):
        ax.add_patch(Circle((x, 4.15), r, facecolor="#dfcdef", edgecolor=edge, lw=1.0))
    ax.plot([5.43, 5.95], [4.15, 4.15], color=edge, lw=1.3)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.plot([x0 + 0.20, x1 - 0.17], [4.15, 4.15], color=edge, lw=0.9)
    ax.text(7.1, 4.63, "photon continuum", ha="center", va="bottom", fontsize=9.5)

    # Observable/source mapping underneath.
    ax.text(2.7, 0.55, r"outputs: $(\mathbf{n}_C,Q)$", ha="center", va="center", fontsize=9.5)
    ax.text(7.3, 0.55, r"sources: $(\mathbf{v}_C,j)$", ha="center", va="center", fontsize=9.5)
    ax.add_patch(FancyArrowPatch((4.15, 0.55), (5.85, 0.55), arrowstyle="<->", mutation_scale=14,
                                 linewidth=1.3, color=edge))
    ax.text(5.0, 0.89, "causal map on the forward image", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout(pad=0.15)
    fig.savefig(outdir / "device_schematic.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outdir / "device_schematic.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
