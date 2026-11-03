"""Generate Fig. 2: photon-induced polarization of the Hubbard dimer."""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import roots_legendre

METHODS = ("oep", "rpa", "irr")
LABELS = {"exact": "Exact", "oep": "Photon OEP", "rpa": "KS photon RPA", "irr": "IRR"}
LINESTYLES = {"exact": "-", "oep": "--", "rpa": ":", "irr": "-."}
PHOTON_CUTOFF = 32

_GL_X, _GL_W = roots_legendre(256)
_QT = (_GL_X + 1) / 2
FREQUENCIES = _QT / (1 - _QT)
FREQUENCY_WEIGHTS = _GL_W / (2 * (1 - _QT) ** 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No data to write")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dimer(U: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    h = np.array([
        [0, -np.sqrt(2), -np.sqrt(2)],
        [-np.sqrt(2), U + b, 0],
        [-np.sqrt(2), 0, U - b],
    ])
    return h, np.diag([0.0, 2.0, -2.0])


def electronic_dimer(U: float, b: float):
    h, m = dimer(U, b)
    energies, states = eigh(h)
    gaps = energies[1:] - energies[0]
    weights = abs(states[:, 0].conj() @ m @ states[:, 1:]) ** 2
    rho = float(np.vdot(states[:, 0], m @ states[:, 0]).real) / 2
    return float(energies[0]), rho, gaps, weights


def kernel_values(rho, gaps, weights):
    """Return the projected KS response and exact electronic kernel at i*xi."""
    r = np.sqrt(1 - rho * rho)
    ds, ws = 2 / r, 2 * r * r
    d1, d2 = gaps
    w1, w2 = weights
    p, q = d1 * w1 + d2 * w2, d1 * w2 + d2 * w1
    aa, cc, pole2 = 1 / (2 * p), d1 * d2 / (2 * q), d1 * d2 * q / p
    bb = aa * (d1 * d1 + d2 * d2 - pole2) - cc
    da, dc = 1 / (2 * ds * ws) - aa, cc - ds / (2 * ws)
    xi2 = FREQUENCIES ** 2
    chi_s = -2 * ds * ws / (xi2 + ds * ds)
    f_hxc = -da * xi2 + dc + bb * xi2 / (xi2 + pole2)
    return chi_s, f_hxc


def photon_at_bias(U, b0, g, omega, method):
    energy, rho, gaps, weights = electronic_dimer(U, b0)
    chi_s, f_hxc = kernel_values(rho, gaps, weights)
    if method != "irr":
        f_hxc = np.zeros_like(chi_s)
    response = chi_s / (1 - chi_s * f_hxc)
    xi2 = FREQUENCIES ** 2
    interaction = 2 * g * g * xi2 / (omega * (xi2 + omega * omega))
    x = -interaction * response
    if np.any(x <= -1):
        raise RuntimeError("Photon-energy logarithm left the real branch")
    integrand = x if method == "oep" else np.log1p(x)
    photon_energy = float(np.dot(FREQUENCY_WEIGHTS, integrand) / (2 * np.pi))
    return energy, rho, photon_energy


def exact_polarization(U: float, b: float, g: float, omega: float) -> float:
    h, m = dimer(U, b)
    _, states = eigh(h)
    shift = float(np.vdot(states[:, 0], m @ states[:, 0]).real)
    dm = m - shift * np.eye(3)
    p = PHOTON_CUTOFF + 1
    a = sparse.diags(np.sqrt(np.arange(1, p)), 1, shape=(p, p))
    total = (
        sparse.kron(sparse.csr_matrix(h + g * g / omega * (dm @ dm)), sparse.eye(p))
        + sparse.kron(sparse.eye(3), sparse.diags(omega * np.arange(p, dtype=float)))
        + g * sparse.kron(sparse.csr_matrix(dm), a + a.T)
    ).tocsr()
    _, states = eigh(total.toarray(), subset_by_index=(0, 0))
    state = states[:, 0].reshape(3, p)
    return float(np.trace((state @ state.conj().T) @ m).real) / 2


def functional_polarization(U, b, g, omega, method):
    def total_energy(b0):
        energy, rho, photon = photon_at_bias(U, b0, g, omega, method)
        return energy + (b - b0) * rho + photon

    biases = np.linspace(-30.0, 30.0, 161)
    energies = np.array([total_energy(b0) for b0 in biases])
    candidates = []
    for i in range(1, len(biases) - 1):
        if energies[i] <= energies[i - 1] and energies[i] <= energies[i + 1]:
            result = minimize_scalar(
                total_energy, bounds=(biases[i - 1], biases[i + 1]),
                method="bounded", options={"xatol": 1e-12},
            )
            if not result.success:
                raise RuntimeError("Dimer energy minimization failed")
            candidates.append((result.fun, result.x))
    if not candidates:
        raise RuntimeError("Dimer minimum was not bracketed")
    _, b0 = min(candidates)
    return electronic_dimer(U, b0)[1]


def plot_reversal(out: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.38))
    for method in ("exact",) + METHODS:
        ax.plot(
            [row["U"] for row in rows], [row["delta_rho_" + method] for row in rows],
            label=LABELS[method], linestyle=LINESTYLES[method], linewidth=1.9,
        )
    ax.axhline(0, linewidth=0.6, linestyle="--")
    ax.set_xlabel(r"Electronic repulsion $U_H/t_h$", fontsize=9)
    ax.set_ylabel(r"Photon-induced polarization $\delta\rho$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False,
        fontsize=7.5, handlelength=2.3, columnspacing=1.2,
    )
    ax.margins(x=0.015, y=0.08)
    fig.tight_layout()
    fig.savefig(out / "polarization_reversal.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/test_2"))
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for U in np.linspace(0, 8, 81):
        b, g, omega = 0.2, 0.1, 1.0
        rho0 = electronic_dimer(U, b)[1]
        row = {
            "U": float(U), "b": b, "g": g, "omega": omega, "rho_zero": rho0,
            "delta_rho_exact": exact_polarization(U, b, g, omega) - rho0,
        }
        for method in METHODS:
            row["delta_rho_" + method] = functional_polarization(U, b, g, omega, method) - rho0
        rows.append(row)
    write_csv(out / "polarization_reversal.csv", rows)
    plot_reversal(out, rows)
    print(f"Fig. 2 complete: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
