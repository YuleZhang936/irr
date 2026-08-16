#!/usr/bin/env python3
"""Generate the joint electronic-photon inverse benchmark data.

Joint analytic-target inverse benchmark for simultaneous electronic/photon regulators.

This replacement benchmark has an exact continuum endpoint on the whole test
interval, not a numerically generated reference trajectory:

    n_*(t) = 1/2,
    Q_*(t) = Q0 + A [1 - cos(nu t)],
    v_*(t) = omega*d*Q_*(t) - d^2/2,

and j_*(t) is given in closed form by the continuum memory integral below.
The electron-photon interaction is retained for every finite regulator as
-g (n_0-1/2) q_tilde with g=omega*d=0.60 != 0; [H_e,n_0] != 0, so this is
neither a factorized electron/photon propagation nor a pure-dephasing model.
The exact endpoint follows from particle-hole times photon-parity symmetry of
the infinite half-filled chain in the co-moving frame.  Finite electronic
regulators intentionally break that protecting symmetry only at the remote
boundary, through one of two bounded termination profiles whose support moves
to infinity.  Finite photon regulators use either Gauss-Legendre or positive
midpoint quadrature for the same positive residual measure.

For every finite (L,M) the target history is imposed by the exact finite
canonical inverse j_M and by solving the electronic force-balance inverse at
every DOP853 stage.  Thus the physical device source is
v_*(t)+delta_v_{L,M}(t).  A nonzero mixed second difference of delta_v across
L and M directly tests that the finite electronic inverse genuinely feels
both regulators even though the continuum target and source are analytic.
The convergence criterion is family-independent approach of v_{L,M} to v_*
and j_M to j_* as L,M grow; finite-grid errors need not decrease monotonically
at every individual step.

The propagation uses the exact co-moving photon frame.  The only electronic
symmetry breaking is the remote boundary termination; without it the finite
half-filled bipartite chain would inherit the exact n_*=1/2 protection and the
old benchmark would again be nearly separable.  The boundary terms are part
of the admissible regulator class and vanish locally in the thermodynamic
limit.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import DOP853
from scipy.sparse import csr_matrix, diags, dok_matrix
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class Parameters:
    hopping: float = 1.0
    selected_frequency: float = 1.20
    dipole: float = 0.50
    q_initial: float = 0.20
    q_amplitude: float = 0.15
    target_frequency: float = 0.50
    bath_strength: float = 0.60
    bath_low: float = 0.75
    bath_high: float = 1.65
    boundary_gate_amplitude: float = 0.20
    final_time: float = 2.0
    time_points: int = 401
    boson_cutoff: int = 4
    rtol: float = 2.0e-9
    atol: float = 2.0e-11


P = Parameters()
L_VALUES = (4, 8, 12)
M_VALUES = (1, 2, 4)
ELECTRON_FAMILIES = ("cosine", "smoothstep")
PHOTON_FAMILIES = ("gauss", "midpoint")


def _fermion_basis(length: int, particles: int) -> tuple[list[int], dict[int, int]]:
    states = [sum(1 << i for i in occ) for occ in combinations(range(length), particles)]
    return states, {state: index for index, state in enumerate(states)}


@lru_cache(maxsize=None)
def _electronic_operators(length: int, p: Parameters) -> dict[str, object]:
    if length < 4 or length % 2:
        raise ValueError("The finite electronic regulator must have an even number of sites >= 4.")
    particles = length // 2
    basis, index = _fermion_basis(length, particles)
    dimension = len(basis)

    hopping = dok_matrix((dimension, dimension), dtype=np.complex128)
    for column, state in enumerate(basis):
        for site in range(length - 1):
            left = (state >> site) & 1
            right = (state >> (site + 1)) & 1
            if left != right:
                moved = state ^ (1 << site) ^ (1 << (site + 1))
                hopping[index[moved], column] += -p.hopping
    hopping = hopping.tocsr()

    # Sites are -L/2+1,...,L/2.  The device is x=0 and the remote
    # termination acts only on the rightmost site x=L/2.
    device_index = length // 2 - 1
    boundary_index = length - 1
    n_device = np.asarray([(state >> device_index) & 1 for state in basis], dtype=float)
    n_boundary = np.asarray([(state >> boundary_index) & 1 for state in basis], dtype=float)
    n_matrix = diags(n_device, format="csr")

    first_commutator = hopping @ n_matrix - n_matrix @ hopping
    k_operator = -(n_matrix @ first_commutator - first_commutator @ n_matrix)
    f_e_operator = -(hopping @ first_commutator - first_commutator @ hopping)

    if dimension <= 350:
        eigenvalues, eigenvectors = np.linalg.eigh(hopping.toarray())
        ground_state = eigenvectors[:, int(np.argmin(eigenvalues))]
    else:
        _, eigenvectors = eigsh(hopping, k=1, which="SA", tol=1.0e-13)
        ground_state = eigenvectors[:, 0]
    ground_state = ground_state / np.linalg.norm(ground_state)

    return {
        "hopping": hopping,
        "basis": basis,
        "ground_state": ground_state,
        "n_device": n_device,
        "n_boundary": n_boundary,
        "n_centered": n_device - 0.5,
        "k_operator": k_operator.tocsr(),
        "f_e_operator": f_e_operator.tocsr(),
    }


def _boson_basis(modes: int, cutoff: int) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    states: list[tuple[int, ...]] = []

    def recurse(prefix: list[int], remaining: int, position: int) -> None:
        if position == modes - 1:
            for occupation in range(remaining + 1):
                states.append(tuple(prefix + [occupation]))
            return
        for occupation in range(remaining + 1):
            recurse(prefix + [occupation], remaining - occupation, position + 1)

    recurse([], cutoff, 0)
    return states, {state: index for index, state in enumerate(states)}


@lru_cache(maxsize=None)
def _photon_regulator(order: int, cutoff: int, family: str, p: Parameters) -> dict[str, object]:
    if family == "gauss":
        nodes, weights = leggauss(order)
        omega = 0.5 * (p.bath_high - p.bath_low) * nodes + 0.5 * (p.bath_high + p.bath_low)
        physical_weights = 0.5 * (p.bath_high - p.bath_low) * weights
    elif family == "midpoint":
        edges = np.linspace(p.bath_low, p.bath_high, order + 1)
        omega = 0.5 * (edges[:-1] + edges[1:])
        physical_weights = np.diff(edges)
    else:
        raise ValueError(f"unknown photon family {family!r}")
    normalization = (
        (p.bath_high**3 - p.bath_low**3) / 3.0
        - p.target_frequency**2 * (p.bath_high - p.bath_low)
    )
    memory_weights = (
        p.bath_strength
        * physical_weights
        * (omega**2 - p.target_frequency**2)
        / normalization
    )
    if np.any(memory_weights <= 0.0):
        raise RuntimeError("The proposed memory measure must be strictly positive at every node.")
    couplings = omega * np.sqrt(memory_weights)

    stiffness = np.diag(
        np.concatenate(
            ([p.selected_frequency**2 + np.sum(couplings**2 / omega**2)], omega**2)
        )
    )
    stiffness[0, 1:] = -couplings
    stiffness[1:, 0] = -couplings
    eigenvalues, eigenvectors = np.linalg.eigh(stiffness)
    if float(eigenvalues.min()) <= 0.0:
        raise RuntimeError("The finite star stiffness is not positive.")
    normal_frequencies = np.sqrt(eigenvalues)
    selected_vector = eigenvectors[0, :]

    basis, basis_index = _boson_basis(order + 1, cutoff)
    dimension = len(basis)
    free_energy = np.asarray(
        [sum(normal_frequencies[k] * state[k] for k in range(order + 1)) for state in basis],
        dtype=float,
    )
    selected_q = dok_matrix((dimension, dimension), dtype=np.complex128)
    for column, state in enumerate(basis):
        total = sum(state)
        for mode in range(order + 1):
            coefficient = selected_vector[mode] / math.sqrt(2.0 * normal_frequencies[mode])
            if state[mode] > 0:
                lowered = list(state)
                lowered[mode] -= 1
                selected_q[basis_index[tuple(lowered)], column] += coefficient * math.sqrt(state[mode])
            if total < cutoff:
                raised = list(state)
                raised[mode] += 1
                selected_q[basis_index[tuple(raised)], column] += coefficient * math.sqrt(state[mode] + 1)

    top_sector = np.asarray([sum(state) == cutoff for state in basis], dtype=bool)
    return {
        "bath_frequencies": omega,
        "physical_weights": physical_weights,
        "couplings": couplings,
        "normalization": normalization,
        "stiffness_eigenvalues": eigenvalues,
        "normal_frequencies": normal_frequencies,
        "free_energy": free_energy,
        "selected_q": selected_q.tocsr(),
        "basis": basis,
        "top_sector": top_sector,
    }


def _boundary_gate(time_value: float, family: str, p: Parameters) -> float:
    x = min(max(time_value / p.final_time, 0.0), 1.0)
    if family == "cosine":
        return 0.5 * p.boundary_gate_amplitude * (1.0 - math.cos(math.pi * x))
    if family == "smoothstep":
        return p.boundary_gate_amplitude * (3.0 * x**2 - 2.0 * x**3)
    raise ValueError(f"unknown electronic family {family!r}")


def _target_functions(times: np.ndarray, p: Parameters) -> dict[str, np.ndarray]:
    q = p.q_initial + p.q_amplitude * (1.0 - np.cos(p.target_frequency * times))
    qdd = p.q_amplitude * p.target_frequency**2 * np.cos(p.target_frequency * times)
    v_star = p.selected_frequency * p.dipole * q - 0.5 * p.dipole**2
    return {"q": q, "qdd": qdd, "v_star": v_star}


def _photon_sources(order: int, times: np.ndarray, regulator: dict[str, object], p: Parameters) -> dict[str, np.ndarray | float]:
    target = _target_functions(times, p)
    omega = np.asarray(regulator["bath_frequencies"], dtype=float)
    weights = np.asarray(regulator["physical_weights"], dtype=float)
    normalization = float(regulator["normalization"])

    residual_finite = (
        p.q_amplitude
        * p.target_frequency**2
        * p.bath_strength
        / normalization
        * (
            np.sum(weights[:, None] * np.cos(omega[:, None] * times[None, :]), axis=0)
            - (p.bath_high - p.bath_low) * np.cos(p.target_frequency * times)
        )
    )
    continuum_cosine = np.empty_like(times)
    continuum_cosine[0] = p.bath_high - p.bath_low
    nonzero = times != 0.0
    continuum_cosine[nonzero] = (
        np.sin(p.bath_high * times[nonzero]) - np.sin(p.bath_low * times[nonzero])
    ) / times[nonzero]
    residual_infinite = (
        p.q_amplitude
        * p.target_frequency**2
        * p.bath_strength
        / normalization
        * (
            continuum_cosine
            - (p.bath_high - p.bath_low) * np.cos(p.target_frequency * times)
        )
    )
    common = (
        target["qdd"]
        + p.selected_frequency**2 * target["q"]
        - 0.5 * p.selected_frequency * p.dipole
    )
    j_finite = common - residual_finite
    j_infinite = common - residual_infinite
    error = np.abs(j_finite - j_infinite)
    return {
        "j_finite": j_finite,
        "j_infinite": j_infinite,
        "residual_finite": residual_finite,
        "residual_infinite": residual_infinite,
        "sampled_error": float(np.max(error)),
        "sampled_argmax": float(times[int(np.argmax(error))]),
    }


def _stream_dop853(
    fun,
    y0: np.ndarray,
    times: np.ndarray,
    rtol: float,
    atol: float,
) -> Iterable[tuple[float, np.ndarray]]:
    solver = DOP853(fun, float(times[0]), y0, float(times[-1]), rtol=rtol, atol=atol)
    next_index = 0
    yield float(times[0]), y0.copy()
    next_index = 1
    while solver.status == "running":
        previous_time = solver.t
        solver.step()
        if solver.status == "failed":
            raise RuntimeError("DOP853 propagation failed.")
        interpolant = solver.dense_output()
        while next_index < len(times) and times[next_index] <= solver.t + 10.0 * np.finfo(float).eps:
            time_value = float(times[next_index])
            if time_value < previous_time - 10.0 * np.finfo(float).eps:
                raise RuntimeError("Internal time-grid ordering failure.")
            yield time_value, np.asarray(interpolant(time_value), dtype=np.complex128)
            next_index += 1
    if next_index != len(times):
        raise RuntimeError("The streaming integrator did not fill the requested time grid.")


def run_pair(
    length: int,
    order: int,
    electron_family: str,
    photon_family: str,
    *,
    cutoff: int | None = None,
    p: Parameters = P,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, object]:
    cutoff = p.boson_cutoff if cutoff is None else cutoff
    rtol = p.rtol if rtol is None else rtol
    atol = p.atol if atol is None else atol
    if electron_family not in ELECTRON_FAMILIES:
        raise ValueError(f"unknown electronic family {electron_family!r}")
    if photon_family not in PHOTON_FAMILIES:
        raise ValueError(f"unknown photon family {photon_family!r}")
    electron = _electronic_operators(length, p)
    photon = _photon_regulator(order, cutoff, photon_family, p)

    h_e: csr_matrix = electron["hopping"]  # type: ignore[assignment]
    k_e: csr_matrix = electron["k_operator"]  # type: ignore[assignment]
    f_e: csr_matrix = electron["f_e_operator"]  # type: ignore[assignment]
    q: csr_matrix = photon["selected_q"]  # type: ignore[assignment]
    q_transpose = q.transpose().tocsr()
    free_energy = np.asarray(photon["free_energy"], dtype=float)

    ground_state = np.asarray(electron["ground_state"], dtype=np.complex128)
    n_device = np.asarray(electron["n_device"], dtype=float)
    n_boundary = np.asarray(electron["n_boundary"], dtype=float)
    n_centered = np.asarray(electron["n_centered"], dtype=float)
    electronic_dimension = len(ground_state)
    boson_dimension = len(free_energy)
    initial = np.zeros((electronic_dimension, boson_dimension), dtype=np.complex128)
    initial[:, 0] = ground_state

    coupling = p.selected_frequency * p.dipole
    n_device_column = n_device[:, None]
    n_boundary_column = n_boundary[:, None]
    n_centered_column = n_centered[:, None]
    free_energy_row = free_energy[None, :]

    def source_from_state(state_matrix: np.ndarray) -> tuple[float, float, float]:
        # Normalize all expectation values.  The reconstructed ratio -F/K is
        # invariant under an overall norm drift, but K and F themselves are
        # physical expectation values and should be reported normalized.
        norm_value = float(np.vdot(state_matrix, state_matrix).real)
        if not np.isfinite(norm_value) or norm_value <= 0.0:
            raise RuntimeError("Invalid numerical-state norm.")
        k_state = k_e @ state_matrix
        k_value = float(np.vdot(state_matrix, k_state).real) / norm_value
        f_state = f_e @ state_matrix - coupling * ((k_state) @ q_transpose)
        f_value = float(np.vdot(state_matrix, f_state).real) / norm_value
        if not np.isfinite(k_value) or abs(k_value) < 0.25:
            raise RuntimeError(f"Device inverse lost conditioning: K={k_value:.6e}")
        delta_v = -f_value / k_value
        return delta_v, k_value, f_value

    def rhs(time_value: float, flat_state: np.ndarray) -> np.ndarray:
        # Exact co-moving frame: v_*(t) and j_M(t) generate the prescribed
        # classical trajectory and therefore do not appear explicitly here.
        # delta_v is the finite-regulator correction reconstructed from the
        # force-balance identity; the physical source is v_* + delta_v.
        state = flat_state.reshape(electronic_dimension, boson_dimension)
        delta_v, _, _ = source_from_state(state)
        derivative = h_e @ state
        derivative = derivative + state * free_energy_row
        derivative = derivative - coupling * n_centered_column * (state @ q_transpose)
        derivative = derivative + _boundary_gate(time_value, electron_family, p) * n_boundary_column * state
        derivative = derivative + delta_v * n_device_column * state
        return (-1.0j * derivative).reshape(-1)

    times = np.linspace(0.0, p.final_time, p.time_points)
    delta_v_trace = np.empty_like(times)
    gap_trace = np.empty_like(times)
    density_trace = np.empty_like(times)
    q_centered_trace = np.empty_like(times)
    norm_trace = np.empty_like(times)
    top_trace = np.empty_like(times)
    top_sector = np.asarray(photon["top_sector"], dtype=bool)

    for index, (time_value, flat_state) in enumerate(
        _stream_dop853(rhs, initial.reshape(-1), times, rtol, atol)
    ):
        state = flat_state.reshape(electronic_dimension, boson_dimension)
        delta_v, k_value, _ = source_from_state(state)
        probabilities = np.abs(state) ** 2
        norm_value = float(np.vdot(flat_state, flat_state).real)
        delta_v_trace[index] = delta_v
        gap_trace[index] = -k_value
        density_trace[index] = float(np.sum(probabilities * n_device_column) / norm_value)
        q_centered_trace[index] = float(np.vdot(state, state @ q_transpose).real / norm_value)
        norm_trace[index] = norm_value
        top_trace[index] = float(np.sum(probabilities[:, top_sector]) / norm_value)

    photon_sources = _photon_sources(order, times, photon, p)
    target = _target_functions(times, p)
    physical_v = target["v_star"] + delta_v_trace
    e_v_trace = np.abs(physical_v - target["v_star"])
    e_j_trace = np.abs(
        np.asarray(photon_sources["j_finite"]) - np.asarray(photon_sources["j_infinite"])
    )

    result = {
        "electronic_family": electron_family,
        "photon_family": photon_family,
        "L": length,
        "M": order,
        "boson_cutoff": cutoff,
        "electronic_dimension": electronic_dimension,
        "boson_dimension": boson_dimension,
        "total_dimension": electronic_dimension * boson_dimension,
        "E_v": float(np.max(e_v_trace)),
        "E_j": float(np.max(e_j_trace)),
        "E_src": float(max(np.max(e_v_trace), np.max(e_j_trace))),
        "argmax_v": float(times[int(np.argmax(e_v_trace))]),
        "argmax_j": float(times[int(np.argmax(e_j_trace))]),
        "min_kappa": float(np.min(gap_trace)),
        "max_density_tracking_error": float(np.max(np.abs(density_trace - 0.5))),
        "max_centered_q_error": float(np.max(np.abs(q_centered_trace))),
        "max_norm_error": float(np.max(np.abs(norm_trace - 1.0))),
        "max_top_sector_population": float(np.max(top_trace)),
        "stiffness_min": float(np.min(np.asarray(photon["stiffness_eigenvalues"]))),
        "stiffness_max": float(np.max(np.asarray(photon["stiffness_eigenvalues"]))),
        "time_grid_points": len(times),
        "time_grid_step": float(times[1] - times[0]),
        "times": times,
        "delta_v_trace": delta_v_trace,
        "density_trace": density_trace,
        "q_centered_trace": q_centered_trace,
        "norm_trace": norm_trace,
        "top_trace": top_trace,
        "v_star_trace": np.asarray(target["v_star"], dtype=float),
        "v_physical_trace": np.asarray(physical_v, dtype=float),
        "j_finite_trace": np.asarray(photon_sources["j_finite"], dtype=float),
        "j_infinite_trace": np.asarray(photon_sources["j_infinite"], dtype=float),
        "e_v_trace": e_v_trace,
        "e_j_trace": e_j_trace,
    }
    return result


def _tex_number(value: float) -> str:
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _serializable(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    primary = [r for r in rows if r["electronic_family"] == "cosine" and r["photon_family"] == "gauss"]
    primary.sort(key=lambda r: (int(r["L"]), int(r["M"])))
    lines = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$L$ & $M$ & $E^v_{L,M}$ & $E^j_M$ & $E^{\rm src}_{L,M}$\\",
        r"\midrule",
    ]
    for row in primary:
        lines.append(
            rf"{row['L']} & {row['M']} & ${_tex_number(float(row['E_v']))}$ & "
            rf"${_tex_number(float(row['E_j']))}$ & ${_tex_number(float(row['E_src']))}$\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _primary(rows: list[dict[str, object]], L: int, M: int) -> dict[str, object]:
    return next(
        r for r in rows
        if r["electronic_family"] == "cosine" and r["photon_family"] == "gauss"
        and int(r["L"]) == L and int(r["M"]) == M
    )


def _write_macros(path: Path, rows: list[dict[str, object]], cross_rows: list[dict[str, object]], checks: dict[str, object]) -> None:
    near = _primary(rows, min(L_VALUES), min(M_VALUES))
    far = _primary(rows, max(L_VALUES), max(M_VALUES))
    primary_cross = next(r for r in cross_rows if r["electronic_family"] == "cosine" and r["photon_family"] == "gauss")
    e_spreads = checks["electronic_family_spreads"]
    p_spreads = checks["photon_family_spreads"]
    content = "".join([
        f"\\newcommand{{\\AnalyticJointNear}}{{\\ensuremath{{{_tex_number(float(near['E_src']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointFar}}{{\\ensuremath{{{_tex_number(float(far['E_src']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointCross}}{{\\ensuremath{{{_tex_number(float(primary_cross['cross_v']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointElectronSpreadCoarse}}{{\\ensuremath{{{_tex_number(float(e_spreads[0]['spread_v']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointElectronSpreadFine}}{{\\ensuremath{{{_tex_number(float(e_spreads[-1]['spread_v']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointPhotonSpreadFineV}}{{\\ensuremath{{{_tex_number(float(p_spreads[-1]['spread_v']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointPhotonSpreadFineJ}}{{\\ensuremath{{{_tex_number(float(p_spreads[-1]['spread_j']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointKappaMin}}{{\\ensuremath{{{float(min(float(r['min_kappa']) for r in rows)):.3f}}}}}\n",
        f"\\newcommand{{\\AnalyticJointDensityMax}}{{\\ensuremath{{{_tex_number(float(max(float(r['max_density_tracking_error']) for r in rows)))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointQMax}}{{\\ensuremath{{{_tex_number(float(max(float(r['max_centered_q_error']) for r in rows)))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointNormMax}}{{\\ensuremath{{{_tex_number(float(max(float(r['max_norm_error']) for r in rows)))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointFockMax}}{{\\ensuremath{{{_tex_number(float(max(float(r['max_top_sector_population']) for r in rows)))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointCutoffV}}{{\\ensuremath{{{_tex_number(float(checks['cutoff_check']['max_v_trace_difference']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointTightV}}{{\\ensuremath{{{_tex_number(float(checks['tolerance_check']['max_v_trace_difference']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointCrossCutoffV}}{{\\ensuremath{{{_tex_number(float(checks['mixed_defect_numerical_floor']['max_cutoff_v_trace_difference']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointCrossTightV}}{{\\ensuremath{{{_tex_number(float(checks['mixed_defect_numerical_floor']['max_tight_v_trace_difference']))}}}}}\n",
        f"\\newcommand{{\\AnalyticJointCrossFloor}}{{\\ensuremath{{{_tex_number(float(checks['mixed_defect_numerical_floor']['max_numerical_floor']))}}}}}\n",
    ])
    path.write_text(content, encoding="utf-8")


def _plot_heatmap(path_pdf: Path, path_png: Path, rows: list[dict[str, object]]) -> None:
    values = np.asarray([
        [float(_primary(rows, L, M)["E_src"]) for L in L_VALUES]
        for M in M_VALUES
    ])
    fig, axis = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    image = axis.imshow(np.log10(values), origin="lower", aspect="auto")
    axis.set_xticks(np.arange(len(L_VALUES)), [str(v) for v in L_VALUES])
    axis.set_yticks(np.arange(len(M_VALUES)), [str(v) for v in M_VALUES])
    axis.set_xlabel(r"electronic regulator $L$")
    axis.set_ylabel(r"photon regulator $M$")
    axis.tick_params(labelsize=7.5)
    axis.xaxis.label.set_size(8.5); axis.yaxis.label.set_size(8.5)
    threshold = float(np.nanmean(np.log10(values)))
    for i in range(len(M_VALUES)):
        for k in range(len(L_VALUES)):
            val = values[i, k]
            axis.text(k, i, f"{val:.1e}", ha="center", va="center", fontsize=6.2,
                      color="white" if np.log10(val) < threshold else "black")
    cb = fig.colorbar(image, ax=axis, pad=0.02)
    cb.set_label(r"$\log_{10} E^{\rm src}_{L,M}$", fontsize=8.2)
    cb.ax.tick_params(labelsize=7.0)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _estimated_dimension(length: int, order: int, cutoff: int) -> int:
    return math.comb(length, length // 2) * math.comb(cutoff + order + 1, order + 1)


def _run_task(task: tuple[str, str, int, int, Parameters]) -> tuple[tuple[str, str, int, int], dict[str, object]]:
    ef, pf, L, M, p = task
    return (ef, pf, L, M), run_pair(L, M, ef, pf, p=p)


def _trace_map(results: dict[tuple[str, str, int, int], dict[str, object]], key: tuple[str, str, int, int]) -> np.ndarray:
    return np.asarray(results[key]["delta_v_trace"], dtype=float)


def _save_result_npz(path: Path, result: dict[str, object]) -> None:
    scalars={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(path, metadata=json.dumps(scalars), **arrays)


def _load_result_npz(path: Path) -> dict[str, object]:
    with np.load(path,allow_pickle=False) as z:
        result=json.loads(str(z["metadata"]))
        for k in z.files:
            if k!="metadata": result[k]=np.asarray(z[k])
    return result


def _fresh_run_pair(output_dir: Path, L: int, M: int, ef: str, pf: str, p: Parameters, *, cutoff: int, rtol: float, atol: float) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="analytic_joint_fresh_", dir=output_dir) as td:
        td=Path(td); out=td/"case.npz"; par=td/"params.json"
        par.write_text(json.dumps(asdict(p)),encoding="utf-8")
        env=dict(**__import__("os").environ)
        env.setdefault("OMP_NUM_THREADS","1"); env.setdefault("OPENBLAS_NUM_THREADS","1"); env.setdefault("MKL_NUM_THREADS","1")
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"--single-case-out",str(out),"--params-json",str(par),
                        "--L",str(L),"--M",str(M),"--electron-family",ef,"--photon-family",pf,
                        "--cutoff",str(cutoff),"--case-rtol",repr(rtol),"--case-atol",repr(atol)],check=True,env=env)
        return _load_result_npz(out)


def run_all(output_dir: Path, figure_dir: Path, p: Parameters = P) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    # Run numerical-floor checks at every corner entering the mixed defect.
    # A four-trace mixed defect must not be certified from the fine point alone.
    print("[benchmark] starting four-corner numerical-floor checks", flush=True)
    mixed_corners=((min(L_VALUES),min(M_VALUES)),(min(L_VALUES),max(M_VALUES)),
                   (max(L_VALUES),min(M_VALUES)),(max(L_VALUES),max(M_VALUES)))
    corner_preflight={}
    for Lc,Mc in mixed_corners:
        base_check = _fresh_run_pair(output_dir,Lc,Mc,"cosine","gauss",p,cutoff=p.boson_cutoff,rtol=p.rtol,atol=p.atol)
        cutoff_check_run = _fresh_run_pair(output_dir,Lc,Mc,"cosine","gauss",p,cutoff=p.boson_cutoff+1,rtol=p.rtol,atol=p.atol)
        tight_check_run = _fresh_run_pair(output_dir,Lc,Mc,"cosine","gauss",p,cutoff=p.boson_cutoff,rtol=p.rtol/4.0,atol=p.atol/4.0)
        corner_preflight[Lc,Mc]=(base_check,cutoff_check_run,tight_check_run)
    print("[benchmark] four-corner checks complete", flush=True)

    rows: list[dict[str, object]] = []
    results: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for ef in ELECTRON_FAMILIES:
        for pf in PHOTON_FAMILIES:
            print(f"[benchmark] production family {ef}/{pf}", flush=True)
            for L in L_VALUES:
                for M in M_VALUES:
                    result = run_pair(L, M, ef, pf, p=p)
                    results[(ef, pf, L, M)] = result
                    rows.append({k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
                    gc.collect()
            # Keep only one regulator family of sparse operators resident at a time.
            _photon_regulator.cache_clear(); _electronic_operators.cache_clear(); gc.collect()
    rows.sort(key=lambda r: (str(r["electronic_family"]), str(r["photon_family"]), int(r["L"]), int(r["M"])))
    print("[benchmark] production complete", flush=True)

    fields = ["electronic_family","photon_family","L","M","E_v","E_j","E_src","argmax_v","argmax_j",
              "min_kappa","max_density_tracking_error","max_centered_q_error","max_norm_error",
              "max_top_sector_population","electronic_dimension","boson_dimension","total_dimension",
              "boson_cutoff","stiffness_min","stiffness_max","time_grid_points","time_grid_step"]
    _write_csv(output_dir / "joint_analytic_inverse_grid.csv", rows, fields)
    _write_table(output_dir / "joint_analytic_inverse_table.tex", rows)
    _plot_heatmap(figure_dir / "joint_analytic_inverse.pdf", figure_dir / "joint_analytic_inverse.png", rows)
    print("[benchmark] outputs/plot complete", flush=True)

    cross_rows: list[dict[str, object]] = []
    L0, L1 = min(L_VALUES), max(L_VALUES); M0, M1 = min(M_VALUES), max(M_VALUES)
    for ef in ELECTRON_FAMILIES:
        for pf in PHOTON_FAMILIES:
            cross = _trace_map(results,(ef,pf,L0,M0)) - _trace_map(results,(ef,pf,L0,M1)) \
                  - _trace_map(results,(ef,pf,L1,M0)) + _trace_map(results,(ef,pf,L1,M1))
            cross_rows.append({"electronic_family":ef,"photon_family":pf,"L0":L0,"L1":L1,"M0":M0,"M1":M1,
                               "cross_v":float(np.max(np.abs(cross)))})
    print("[benchmark] cross defects complete", flush=True)
    _write_csv(output_dir / "joint_analytic_inverse_cross.csv", cross_rows,
               ["electronic_family","photon_family","L0","L1","M0","M1","cross_v"])

    electronic_family_spreads=[]
    for L in L_VALUES:
        a=_trace_map(results,("cosine","gauss",L,max(M_VALUES))); b=_trace_map(results,("smoothstep","gauss",L,max(M_VALUES)))
        electronic_family_spreads.append({"L":L,"M":max(M_VALUES),"spread_v":float(np.max(np.abs(a-b)))})
    photon_family_spreads=[]
    for M in M_VALUES:
        a=_trace_map(results,("cosine","gauss",max(L_VALUES),M)); b=_trace_map(results,("cosine","midpoint",max(L_VALUES),M))
        ja=results[("cosine","gauss",max(L_VALUES),M)]; jb=results[("cosine","midpoint",max(L_VALUES),M)]
        # j traces are represented by absolute errors only in result; reconstruct exact family spread directly.
        times=np.asarray(ja["times"],dtype=float)
        pga=_photon_sources(M,times,_photon_regulator(M,p.boson_cutoff,"gauss",p),p)
        pmi=_photon_sources(M,times,_photon_regulator(M,p.boson_cutoff,"midpoint",p),p)
        photon_family_spreads.append({"L":max(L_VALUES),"M":M,
            "spread_v":float(np.max(np.abs(a-b))),
            "spread_j":float(np.max(np.abs(np.asarray(pga["j_finite"])-np.asarray(pmi["j_finite"]))))})

    corner_checks=[]
    for Lc,Mc in ((L0,M0),(L0,M1),(L1,M0),(L1,M1)):
        base_check,cutoff_check_run,tight_check_run=corner_preflight[Lc,Mc]
        corner_checks.append({
            "L":Lc,"M":Mc,
            "cutoff_max_v_trace_difference":float(np.max(np.abs(np.asarray(base_check["delta_v_trace"])-np.asarray(cutoff_check_run["delta_v_trace"])))),
            "cutoff_density_trace_difference":float(np.max(np.abs(np.asarray(base_check["density_trace"])-np.asarray(cutoff_check_run["density_trace"])))),
            "top_population_Nmax":float(base_check["max_top_sector_population"]),
            "top_population_Nmax_check":float(cutoff_check_run["max_top_sector_population"]),
            "tight_max_v_trace_difference":float(np.max(np.abs(np.asarray(base_check["delta_v_trace"])-np.asarray(tight_check_run["delta_v_trace"])))),
        })
    cross_cutoff=max(c["cutoff_max_v_trace_difference"] for c in corner_checks)
    cross_tight=max(c["tight_max_v_trace_difference"] for c in corner_checks)
    cross_floor=max(cross_cutoff,cross_tight)
    fine_corner=next(c for c in corner_checks if c["L"]==L1 and c["M"]==M1)
    cutoff_check={
        "L":L1,"M":M1,"Nmax":p.boson_cutoff,"Nmax_check":p.boson_cutoff+1,
        "max_v_trace_difference":fine_corner["cutoff_max_v_trace_difference"],
        "density_trace_difference":fine_corner["cutoff_density_trace_difference"],
        "top_population_Nmax":fine_corner["top_population_Nmax"],
        "top_population_Nmax_check":fine_corner["top_population_Nmax_check"],
    }
    tolerance_check={
        "L":L1,"M":M1,"production_rtol":p.rtol,"tight_rtol":p.rtol/4.0,
        "max_v_trace_difference":fine_corner["tight_max_v_trace_difference"],
    }
    mixed_floor_check={
        "corners":corner_checks,
        "max_cutoff_v_trace_difference":cross_cutoff,
        "max_tight_v_trace_difference":cross_tight,
        "max_numerical_floor":cross_floor,
    }
    checks={"electronic_family_spreads":electronic_family_spreads,"photon_family_spreads":photon_family_spreads,
            "cutoff_check":cutoff_check,"tolerance_check":tolerance_check,
            "mixed_defect_numerical_floor":mixed_floor_check}
    _write_macros(output_dir / "joint_analytic_inverse_macros.tex", rows, cross_rows, checks)

    # Acceptance conditions: exact-target tracking, safe inverse gap, resolved cross signal and numerical floor.
    if min(float(r["min_kappa"]) for r in rows) <= 0.75:
        raise RuntimeError("inverse gap became too small")
    if max(float(r["max_density_tracking_error"]) for r in rows) >= 2.0e-9:
        raise RuntimeError("analytic density target was not tracked accurately")
    if max(float(r["max_centered_q_error"]) for r in rows) >= 2.0e-8:
        raise RuntimeError("analytic photon target was not tracked accurately")
    if max(float(r["max_norm_error"]) for r in rows) >= 2.0e-7:
        raise RuntimeError("norm drift too large")
    if max(float(r["max_top_sector_population"]) for r in rows) >= 5.0e-6:
        raise RuntimeError("production Fock cutoff not converged")
    primary_cross=float(next(r["cross_v"] for r in cross_rows if r["electronic_family"]=="cosine" and r["photon_family"]=="gauss"))
    if primary_cross <= 1000.0*float(checks["mixed_defect_numerical_floor"]["max_numerical_floor"]):
        raise RuntimeError("mixed L-M signal is not separated from the four-corner numerical floor")
    near=_primary(rows,min(L_VALUES),min(M_VALUES)); far=_primary(rows,max(L_VALUES),max(M_VALUES))
    if float(far["E_src"]) >= 0.12*float(near["E_src"]):
        raise RuntimeError("fine analytic-endpoint source error did not clearly improve")

    payload={
        "benchmark":"joint_analytic_inverse_replacement",
        "endpoint":"exact closed-form continuum target and source; no numerical reference trajectory",
        "analytic_target":{
            "n":"1/2",
            "Q":"Q0 + A*(1-cos(nu*t))",
            "v":"omega*d*Q - d^2/2",
            "j":"Qdd + omega^2 Q - omega*d/2 - R_infty(t)",
            "R_infty":"A*nu^2*eta/Z * [(sin(Omega_plus t)-sin(Omega_minus t))/t - (Omega_plus-Omega_minus) cos(nu t)] with removable t=0 value 0",
        },
        "metadata":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__},
        "parameters":asdict(p),"L_values":list(L_VALUES),"M_values":list(M_VALUES),
        "electronic_families":list(ELECTRON_FAMILIES),"photon_families":list(PHOTON_FAMILIES),
        "grid":rows,"cross_defects":cross_rows,"checks":checks,
    }
    (output_dir / "joint_analytic_inverse_benchmark.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("results"))
    parser.add_argument("--figure-dir",type=Path,default=Path("figures"))
    parser.add_argument("--single-case-out",type=Path)
    parser.add_argument("--worker-family-dir",type=Path)
    parser.add_argument("--params-json",type=Path)
    parser.add_argument("--L",type=int); parser.add_argument("--M",type=int)
    parser.add_argument("--electron-family"); parser.add_argument("--photon-family")
    parser.add_argument("--cutoff",type=int); parser.add_argument("--case-rtol",type=float); parser.add_argument("--case-atol",type=float)
    args=parser.parse_args()
    if args.single_case_out is not None:
        if args.params_json is None: case_p=P
        else: case_p=Parameters(**json.loads(args.params_json.read_text(encoding="utf-8")))
        result=run_pair(args.L,args.M,args.electron_family,args.photon_family,cutoff=args.cutoff,p=case_p,rtol=args.case_rtol,atol=args.case_atol)
        _save_result_npz(args.single_case_out,result)
        return 0
    if args.worker_family_dir is not None:
        if args.params_json is None: case_p=P
        else: case_p=Parameters(**json.loads(args.params_json.read_text(encoding="utf-8")))
        args.worker_family_dir.mkdir(parents=True, exist_ok=True)
        if args.electron_family not in ELECTRON_FAMILIES or args.photon_family not in PHOTON_FAMILIES:
            raise SystemExit("worker requires valid --electron-family and --photon-family")
        for L in L_VALUES:
            for M in M_VALUES:
                print(f"[worker] {args.electron_family}/{args.photon_family} L={L} M={M}", flush=True)
                result=run_pair(L,M,args.electron_family,args.photon_family,p=case_p)
                _save_result_npz(args.worker_family_dir / f"{args.electron_family}_{args.photon_family}_L{L}_M{M}.npz", result)
        return 0
    payload=run_all(args.output_dir,args.figure_dir)
    print(json.dumps({"grid":payload["grid"],"cross_defects":payload["cross_defects"],"checks":payload["checks"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
