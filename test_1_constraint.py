"""Generate Fig. 1: weak-coupling errors in energy, potential and transferred electrons."""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
import torch

from test_2_ground_state import (
    METHODS, LABELS, LINESTYLES, PHOTON_CUTOFF, FREQUENCIES, FREQUENCY_WEIGHTS,
    dimer, electronic_dimer, plt, write_csv,
)
from test_3_real_time import initial_system, calculate_case

torch.set_default_dtype(torch.float64)


def fixed_density(g: float) -> list[dict]:
    U, rho, omega = 2.0, 0.25, 1.0
    b0 = brentq(lambda b: electronic_dimer(U, b)[1] - rho, -1000.0, 1000.0, xtol=5e-14)
    h0, m = dimer(U, 0.0)
    b = torch.tensor(b0, requires_grad=True)
    h = torch.tensor(h0) + b * torch.diag(torch.tensor([0.0, 1.0, -1.0]))
    mt = torch.tensor(m)
    energy, states = torch.linalg.eigh(h)
    density = (states[:, 0] @ mt @ states[:, 0]) / 2
    density_derivative = torch.autograd.grad(density, b, retain_graph=True)[0]
    gaps = energy[1:] - energy[0]
    weights = (states[:, 0] @ mt @ states[:, 1:]) ** 2
    z = torch.sqrt(1 - density * density)
    ds, ws = 2 / z, 2 * z * z
    p = (gaps * weights).sum()
    q = gaps[0] * weights[1] + gaps[1] * weights[0]
    aa, cc, pole2 = 1 / (2 * p), gaps.prod() / (2 * q), gaps.prod() * q / p
    bb = aa * ((gaps * gaps).sum() - pole2) - cc
    da, dc = 1 / (2 * ds * ws) - aa, cc - ds / (2 * ws)
    xi, quadrature = torch.tensor(FREQUENCIES), torch.tensor(FREQUENCY_WEIGHTS)
    chi_s = -2 * ds * ws / (xi * xi + ds * ds)
    kernel = -da * xi * xi + dc + bb * xi * xi / (xi * xi + pole2)
    response = chi_s / (1 - chi_s * kernel)
    interaction = 2 * g * g * xi * xi / (omega * (xi * xi + omega * omega))
    irr_energy = (quadrature * torch.log1p(-interaction * response)).sum() / (2 * np.pi)
    irr_derivative = torch.autograd.grad(irr_energy, b)[0]
    approximations = {
        "irr": (float(irr_energy.detach()), float((irr_derivative / density_derivative).detach()))
    }
    for method in ("oep", "rpa"):
        r = torch.tensor(rho, requires_grad=True)
        z = torch.sqrt(1 - r * r)
        gap, weight = 2 / z, 2 * z * z
        if method == "oep":
            value = g * g * gap * weight / (omega * (omega + gap))
        else:
            root = torch.sqrt((omega + gap) ** 2 + 4 * g * g * gap * weight / omega)
            value = 2 * g * g * gap * weight / (omega * (root + omega + gap))
        derivative = torch.autograd.grad(value, r)[0]
        approximations[method] = float(value.detach()), float(derivative.detach())

    p = PHOTON_CUTOFF + 1
    a = np.diag(np.sqrt(np.arange(1, p)), 1)
    total_m = np.kron(m, np.eye(p))

    def exact_at_bias(bias):
        he, _ = dimer(U, bias)
        total = (
            np.kron(he + g * g / omega * (m @ m), np.eye(p))
            + np.kron(np.eye(3), np.diag(omega * np.arange(p)))
            + g * np.kron(m, a + a.T)
        )
        values, vectors = eigh(total)
        return float(values[0]), float(vectors[:, 0] @ total_m @ vectors[:, 0]) / 2

    bg = brentq(lambda bias: exact_at_bias(bias)[1] - rho, -30.0, 30.0, xtol=5e-14)
    exact_energy, _ = exact_at_bias(bg)
    electronic_energy = float(energy[0].detach()) - b0 * rho
    exact = exact_energy - bg * rho - electronic_energy, b0 - bg
    rows = []
    for method in METHODS:
        for k, quantity in enumerate(("energy", "potential")):
            rows.append({
                "g": g, "quantity": quantity, "method": method,
                "exact": exact[k], "approximation": approximations[method][k],
                "error": abs(approximations[method][k] - exact[k]),
            })
    return rows


def fit_powers(rows: list[dict]) -> list[dict]:
    fits = []
    for quantity in ("energy", "potential", "charge"):
        for method in METHODS:
            selected = sorted(
                (r for r in rows if r["quantity"] == quantity and r["method"] == method),
                key=lambda r: r["g"],
            )
            g = np.array([r["g"] for r in selected])
            error = np.array([r["error"] for r in selected])
            if np.any(error <= 0) or not np.all(np.isfinite(error)):
                raise RuntimeError("Power-law fit requires finite positive errors")
            power = float(np.polyfit(np.log(g), np.log(error), 1)[0])
            fits.append({"quantity": quantity, "method": method, "power": power})
    return fits


def scaling_plot(out, rows, fits, quantity, title, filename):
    fig, ax = plt.subplots(figsize=(2.23, 1.72))
    ax.plot([], [])
    powers = []
    for method, marker in zip(METHODS, ("o", "s", "^")):
        selected = sorted(
            (r for r in rows if r["method"] == method and r["quantity"] == quantity),
            key=lambda r: r["g"],
        )
        g = np.array([r["g"] for r in selected])
        error = np.array([r["error"] for r in selected])
        powers.append(next(f["power"] for f in fits
                           if f["quantity"] == quantity and f["method"] == method))
        ax.loglog(g, error, linestyle=LINESTYLES[method], marker=marker,
                  markersize=3.4, linewidth=1.15, label=LABELS[method])
    ax.set_title(title + "\n" + r"$p=" + ", ".join(f"{p:.3f}" for p in powers) + "$",
                 fontsize=8, pad=4)
    ax.set_xlabel(r"$g/t_h$", fontsize=8)
    ax.set_ylabel("Absolute error" if quantity == "charge" else r"Absolute error / $t_h$", fontsize=8)
    ax.set_xticks(g, [f"{value:g}" for value in g], fontsize=7)
    ax.tick_params(axis="both", labelsize=7, pad=2)
    ax.minorticks_off()
    ax.margins(x=0.07, y=0.1)
    fig.tight_layout(pad=0.25)
    fig.savefig(out / (filename + ".pdf"), bbox_inches="tight")
    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)
    return handles, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/test_1"))
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    static = []
    for g in (0.0125, 0.025, 0.05, 0.1):
        static.extend(fixed_density(g))
    write_csv(out / "ground_state_scaling.csv", static)
    e, s = initial_system()
    charge = []
    for g in (0.0025, 0.005, 0.01):
        data = calculate_case(e, s, g, 3.0, tolerance=1e-11)
        for method in METHODS:
            error = float(np.max(abs(data["charge_" + method] - data["charge_exact"])))
            charge.append({"g": g, "quantity": "charge", "method": method, "error": error})
        print(f"Fig. 1(c): g={g:g}", flush=True)
    write_csv(out / "charge_scaling.csv", charge)
    rows = static + charge
    fits = fit_powers(rows)
    write_csv(out / "fitted_exponents.csv", fits)
    for quantity, title, filename in [
        ("energy", "(a) Photon energy", "scaling_functional"),
        ("potential", "(b) Photon potential", "scaling_potential"),
        ("charge", "(c) Transferred electrons", "scaling_charge"),
    ]:
        handles, labels = scaling_plot(out, rows, fits, quantity, title, filename)
    fig = plt.figure(figsize=(6.8, 0.27))
    fig.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=8, handlelength=3)
    fig.savefig(out / "scaling_legend.pdf")
    plt.close(fig)
    print(f"Fig. 1 complete: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
