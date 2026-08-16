#!/usr/bin/env python3
"""Generate the separate electronic and photon benchmark data.

Direct real-time benchmarks with known electronic and photon reservoir limits.

Electronic benchmark
--------------------
A compactly supported one-electron wave packet evolves on the infinite one-dimensional
nearest-neighbour lattice under a prescribed local potential.  The infinite-lattice
target is obtained directly from the exact Bessel propagator and a scalar Volterra
equation, not from a finite reference chain.  Each finite open chain is then propagated
in closed loop: at every adaptive solver stage, the local potential is reconstructed
from the manufactured target acceleration and the current finite state and is fed back
into that finite-chain evolution.  The exact infinite-chain source is used only to
manufacture the target and, after propagation, to score the reconstruction.

Photon benchmark
----------------
An analytic selected-coordinate history is inserted into the exact continuum memory
equation for a compact, gapped Gaussian memory measure.  The continuum source is
computed by adaptive integration over the continuous frequency variable.  Positive
Gauss-Legendre and composite-midpoint rules reconstruct the same source as their order
grows.  No photon-number basis is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
import numpy as np
import scipy
from scipy.integrate import quad, quad_vec, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve
from scipy.special import erf, jv

Array = np.ndarray


@dataclass(frozen=True)
class ElectronReference:
    times: Array
    source: Array
    density: Array
    density_acceleration: Array
    kappa: Array
    psi: dict[int, Array]
    refinement_psi0: float
    refinement_density_acceleration: float


@dataclass(frozen=True)
class FiniteElectronResult:
    sequence: str
    distance: int
    coordinates: Array
    times: Array
    source: Array
    source_error: Array
    density_error: Array
    kappa: Array
    continuous_source_error: float
    continuous_argmax: float
    grid_source_error: float
    kappa_min: float
    kappa_argmin: float
    kappa_grid_min: float
    norm_drift: float
    replay_density_error: float
    replay_state_error: float
    replay_norm_drift: float


@dataclass(frozen=True)
class PhotonReference:
    times: Array
    coordinate: Array
    coordinate_velocity: Array
    coordinate_acceleration: Array
    kernel: Array
    convolution: Array
    source: Array
    quadrature_error_estimate: float


HOPPING = 1.0
INITIAL_SITES = np.arange(-2, 3)
_INITIAL_AMPLITUDES = np.array([0.20, 0.45, 0.70, 0.45, 0.20], dtype=float)
INITIAL_AMPLITUDES = _INITIAL_AMPLITUDES / np.linalg.norm(_INITIAL_AMPLITUDES)


def electronic_source(t: Array | float) -> Array | float:
    return 0.18 * np.sin(0.73 * np.asarray(t)) + 0.055 * np.cos(1.31 * np.asarray(t))


def free_lattice_amplitude(site: int, times: Array) -> Array:
    out = np.zeros_like(times, dtype=np.complex128)
    for initial_site, amplitude in zip(INITIAL_SITES, INITIAL_AMPLITUDES, strict=True):
        order = site - int(initial_site)
        out += (1j ** order) * jv(order, 2.0 * HOPPING * times) * amplitude
    return out


def _direct_infinite_electron(final_time: float, time_points: int) -> tuple[Array, dict[int, Array], Array, Array, Array]:
    if time_points < 3 or (time_points - 1) % 2:
        raise ValueError("time_points must be odd and at least 3")
    times = np.linspace(0.0, final_time, time_points)
    step = float(times[1] - times[0])
    source = np.asarray(electronic_source(times), dtype=float)
    kernel0 = jv(0, 2.0 * HOPPING * times)
    free0 = free_lattice_amplitude(0, times)

    psi0 = np.empty(time_points, dtype=np.complex128)
    forcing = np.empty(time_points, dtype=np.complex128)
    psi0[0] = free0[0]
    forcing[0] = source[0] * psi0[0]
    for index in range(1, time_points):
        interior = np.dot(kernel0[1:index][::-1], forcing[1:index]) if index > 1 else 0.0j
        right_hand_side = free0[index] - 1j * step * (
            0.5 * kernel0[index] * forcing[0] + interior
        )
        psi0[index] = right_hand_side / (1.0 + 0.5j * step * source[index])
        forcing[index] = source[index] * psi0[index]

    amplitudes: dict[int, Array] = {0: psi0}
    for site in (-2, -1, 1, 2):
        free_kernel = (1j ** site) * jv(site, 2.0 * HOPPING * times)
        discrete_convolution = fftconvolve(free_kernel, forcing)[:time_points]
        integral = step * (
            discrete_convolution - 0.5 * free_kernel * forcing[0] - 0.5 * free_kernel[0] * forcing
        )
        amplitudes[site] = free_lattice_amplitude(site, times) - 1j * integral

    psi_c = amplitudes[0]
    internal_acceleration = np.empty(time_points, dtype=float)
    k_matrix = np.empty(time_points, dtype=float)
    for index in range(time_points):
        derivative_under_h0 = 1j * HOPPING * (amplitudes[-1][index] + amplitudes[1][index])
        second_derivative_under_h0 = -(HOPPING**2) * (
            amplitudes[-2][index] + 2.0 * psi_c[index] + amplitudes[2][index]
        )
        internal_acceleration[index] = float(
            2.0 * abs(derivative_under_h0) ** 2
            + 2.0 * np.real(np.conj(psi_c[index]) * second_derivative_under_h0)
        )
        k_matrix[index] = float(
            -2.0 * HOPPING * np.real(np.conj(psi_c[index]) * amplitudes[-1][index])
            -2.0 * HOPPING * np.real(np.conj(psi_c[index]) * amplitudes[1][index])
        )

    # Manufacture the target acceleration directly from the full Schrödinger
    # derivative.  This is algebraically equivalent to F+Kv, but it is evaluated
    # independently of the finite-chain force-balance decomposition later used in
    # the inverse solver.
    hpsi0 = -HOPPING * (amplitudes[-1] + amplitudes[1]) + source * amplitudes[0]
    hpsim1 = -HOPPING * (amplitudes[-2] + amplitudes[0])
    hpsip1 = -HOPPING * (amplitudes[0] + amplitudes[2])
    h2psi0 = -HOPPING * (hpsim1 + hpsip1) + source * hpsi0
    density_acceleration_direct = (
        2.0 * np.abs(hpsi0) ** 2
        - 2.0 * np.real(np.conj(amplitudes[0]) * h2psi0)
    )
    density_acceleration_force_balance = internal_acceleration + k_matrix * source
    identity_error = float(np.max(np.abs(density_acceleration_direct - density_acceleration_force_balance)))
    if identity_error > 5.0e-12:
        raise RuntimeError(f"infinite-target acceleration identity failed: {identity_error:.3e}")
    return times, amplitudes, source, density_acceleration_direct, -k_matrix


def build_electron_reference(final_time: float = 3.0, fine_points: int = 48001) -> ElectronReference:
    fine = _direct_infinite_electron(final_time, fine_points)
    coarse = _direct_infinite_electron(final_time, (fine_points + 1) // 2)
    ratio = (fine_points - 1) // (len(coarse[0]) - 1)
    if ratio != 2:
        raise RuntimeError("unexpected refinement ratio")
    psi_refinement = float(np.max(np.abs(coarse[1][0] - fine[1][0][::ratio])))
    ndd_refinement = float(np.max(np.abs(coarse[3] - fine[3][::ratio])))
    return ElectronReference(
        times=fine[0],
        source=fine[2],
        density=np.abs(fine[1][0]) ** 2,
        density_acceleration=fine[3],
        kappa=fine[4],
        psi=fine[1],
        refinement_psi0=psi_refinement,
        refinement_density_acceleration=ndd_refinement,
    )


def finite_electron_closed_loop(
    reference: ElectronReference,
    coordinates: Array,
    sequence: str,
    distance: int,
    evaluation_points: int = 3001,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-14,
) -> FiniteElectronResult:
    """Solve the finite inverse self-consistently and replay the frozen source.

    The finite right-hand side never receives the exact infinite-chain source.  At every
    DOP853 stage, the force-balance identity reconstructs the potential from the target
    density acceleration and the current normalized finite state.  The exact source is
    called only after the closed-loop solve has finished, to score the reconstruction.
    """
    coordinates = np.asarray(coordinates, dtype=int)
    if 0 not in coordinates:
        raise ValueError("device site 0 must be present")
    site_to_index = {int(site): index for index, site in enumerate(coordinates)}
    device_index = site_to_index[0]
    length = len(coordinates)

    h0 = np.zeros((length, length), dtype=np.complex128)
    for site in coordinates:
        if int(site) + 1 in site_to_index:
            left = site_to_index[int(site)]
            right = site_to_index[int(site) + 1]
            h0[left, right] = h0[right, left] = -HOPPING

    initial_state = np.zeros(length, dtype=np.complex128)
    for site, amplitude in zip(INITIAL_SITES, INITIAL_AMPLITUDES, strict=True):
        if int(site) not in site_to_index:
            raise ValueError("finite volume does not contain the common compact initial support")
        initial_state[site_to_index[int(site)]] = amplitude

    target_ndd = CubicSpline(reference.times, reference.density_acceleration)
    target_density = CubicSpline(reference.times, reference.density)

    def normalized_expectations(state: Array) -> tuple[float, Array, Array]:
        norm2 = float(np.vdot(state, state).real)
        if not np.isfinite(norm2) or norm2 <= 0.0:
            raise RuntimeError(f"invalid state norm: {norm2!r}")
        h0_state = h0 @ state
        return norm2, h0_state, h0 @ h0_state

    def reconstruct(time_value: float, state: Array) -> tuple[float, float]:
        norm2, h0_state, h0_squared_state = normalized_expectations(state)
        first = -1j * h0_state
        second = -h0_squared_state
        internal = float(
            (2.0 * abs(first[device_index]) ** 2
             + 2.0 * np.real(np.conj(state[device_index]) * second[device_index])) / norm2
        )
        k_value = 0.0
        for neighbour in (-1, 1):
            if neighbour in site_to_index:
                neighbour_index = site_to_index[neighbour]
                k_value += 2.0 * np.real(
                    h0[device_index, neighbour_index]
                    * np.conj(state[device_index])
                    * state[neighbour_index]
                ) / norm2
        if abs(k_value) < 1.0e-9:
            raise RuntimeError(f"device matrix became singular at t={time_value:.12g}: K={k_value:.6e}")
        potential = (float(target_ndd(time_value)) - internal) / k_value
        return float(potential), float(-k_value)

    def rhs(time_value: float, state: Array) -> Array:
        potential, _ = reconstruct(time_value, state)
        derivative = h0 @ state
        derivative[device_index] += potential * state[device_index]
        return -1j * derivative

    final_time = float(reference.times[-1])
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_state,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=0.005,
    )
    if not solution.success:
        raise RuntimeError(f"finite closed-loop inverse failed: {solution.message}")

    times = np.linspace(0.0, final_time, evaluation_points)
    source = np.empty(evaluation_points, dtype=float)
    kappa = np.empty(evaluation_points, dtype=float)
    density = np.empty(evaluation_points, dtype=float)
    norm_drift = 0.0
    for index, time_value in enumerate(times):
        state = solution.sol(time_value)
        source[index], kappa[index] = reconstruct(float(time_value), state)
        norm2 = float(np.vdot(state, state).real)
        density[index] = abs(state[device_index]) ** 2 / norm2
        norm_drift = max(norm_drift, abs(norm2 - 1.0))

    exact_source = np.asarray(electronic_source(times), dtype=float)
    source_error = np.abs(source - exact_source)

    # All discrete local maxima on a dense grid are refined.  The result is a
    # dense-interpolant estimate, not an interval-arithmetic global certificate.
    dense_times = np.linspace(0.0, final_time, 6001)

    def residual(time_value: float) -> float:
        state = solution.sol(float(time_value))
        potential, _ = reconstruct(float(time_value), state)
        return abs(potential - float(electronic_source(float(time_value))))

    dense_values = np.asarray([residual(float(t)) for t in dense_times])
    max_candidates = {0, len(dense_times) - 1, int(np.argmax(dense_values))}
    for idx in range(1, len(dense_times) - 1):
        if dense_values[idx] >= dense_values[idx - 1] and dense_values[idx] >= dense_values[idx + 1]:
            max_candidates.add(idx)
    best_value = float(np.max(dense_values))
    best_time = float(dense_times[int(np.argmax(dense_values))])
    dt = float(dense_times[1] - dense_times[0])
    for idx in sorted(max_candidates):
        lo = max(0.0, float(dense_times[idx] - 2.0 * dt))
        hi = min(final_time, float(dense_times[idx] + 2.0 * dt))
        if hi <= lo:
            continue
        opt = minimize_scalar(
            lambda x: -residual(float(x)),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1.0e-13, "maxiter": 200},
        )
        value = -float(opt.fun)
        if value > best_value:
            best_value = value
            best_time = float(opt.x)

    # Refine every local minimum of the inversion scale on the same dense interpolant.
    dense_kappa = np.asarray([reconstruct(float(t), solution.sol(float(t)))[1] for t in dense_times])
    min_candidates: list[tuple[float, float]] = [
        (float(dense_kappa[0]), float(dense_times[0])),
        (float(dense_kappa[-1]), float(dense_times[-1])),
    ]
    for idx in range(1, len(dense_times) - 1):
        if dense_kappa[idx] <= dense_kappa[idx - 1] and dense_kappa[idx] <= dense_kappa[idx + 1]:
            opt = minimize_scalar(
                lambda x: reconstruct(float(x), solution.sol(float(x)))[1],
                bounds=(float(dense_times[idx - 1]), float(dense_times[idx + 1])),
                method="bounded",
                options={"xatol": 1.0e-13, "maxiter": 200},
            )
            min_candidates.append((float(opt.fun), float(opt.x)))
    kappa_min, kappa_argmin = min(min_candidates, key=lambda item: item[0])
    kappa_grid_min = float(np.min(dense_kappa))

    density_error = np.abs(density - target_density(times))

    # Freeze the reconstructed source and replay a standard forward solve whose RHS
    # contains no inverse call.  This round trip is a consistency check, not a second benchmark.
    source_spline = CubicSpline(times, source)

    def replay_rhs(time_value: float, state: Array) -> Array:
        derivative = h0 @ state
        derivative[device_index] += float(source_spline(time_value)) * state[device_index]
        return -1j * derivative

    replay = solve_ivp(
        replay_rhs,
        (0.0, final_time),
        initial_state,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=0.005,
    )
    if not replay.success:
        raise RuntimeError(f"frozen-source replay failed: {replay.message}")

    replay_density_error = 0.0
    replay_state_error = 0.0
    replay_norm_drift = 0.0
    for time_value in times:
        closed_state = solution.sol(float(time_value))
        replay_state = replay.sol(float(time_value))
        replay_norm2 = float(np.vdot(replay_state, replay_state).real)
        replay_density = abs(replay_state[device_index]) ** 2 / replay_norm2
        replay_density_error = max(replay_density_error, abs(replay_density - float(target_density(time_value))))
        replay_state_error = max(replay_state_error, float(np.linalg.norm(replay_state - closed_state)))
        replay_norm_drift = max(replay_norm_drift, abs(replay_norm2 - 1.0))

    return FiniteElectronResult(
        sequence=sequence,
        distance=distance,
        coordinates=coordinates,
        times=times,
        source=source,
        source_error=source_error,
        density_error=density_error,
        kappa=kappa,
        continuous_source_error=best_value,
        continuous_argmax=best_time,
        grid_source_error=float(np.max(dense_values)),
        kappa_min=float(kappa_min),
        kappa_argmin=float(kappa_argmin),
        kappa_grid_min=kappa_grid_min,
        norm_drift=float(norm_drift),
        replay_density_error=float(replay_density_error),
        replay_state_error=float(replay_state_error),
        replay_norm_drift=float(replay_norm_drift),
    )


OMEGA_MIN = 0.58
OMEGA_MAX = 1.82
MEMORY_MASS = 0.035
MEMORY_CENTER = 1.06
MEMORY_WIDTH = 0.31
SELECTED_FREQUENCY = 1.14
Q0 = 0.15
Q1 = 0.07
Q_FREQ_0 = 0.42
Q_FREQ_1 = 2.20


def memory_density(omega: Array | float) -> Array | float:
    normalization = MEMORY_WIDTH * math.sqrt(math.pi / 2.0) * (
        erf((OMEGA_MAX - MEMORY_CENTER) / (math.sqrt(2.0) * MEMORY_WIDTH))
        - erf((OMEGA_MIN - MEMORY_CENTER) / (math.sqrt(2.0) * MEMORY_WIDTH))
    )
    return MEMORY_MASS * np.exp(-0.5 * ((np.asarray(omega) - MEMORY_CENTER) / MEMORY_WIDTH) ** 2) / normalization


def photon_coordinate(times: Array) -> tuple[Array, Array, Array]:
    q = Q0 * np.cos(Q_FREQ_0 * times) + Q1 * np.sin(Q_FREQ_1 * times)
    qd = -Q0 * Q_FREQ_0 * np.sin(Q_FREQ_0 * times) + Q1 * Q_FREQ_1 * np.cos(Q_FREQ_1 * times)
    qdd = -Q0 * Q_FREQ_0**2 * np.cos(Q_FREQ_0 * times) - Q1 * Q_FREQ_1**2 * np.sin(Q_FREQ_1 * times)
    return q, qd, qdd


def _cos_cos_convolution(omega: Array, frequency: float, times: Array) -> Array:
    return (omega * np.sin(omega * times) - frequency * np.sin(frequency * times)) / (omega**2 - frequency**2)


def _cos_sin_convolution(omega: Array, frequency: float, times: Array) -> Array:
    return frequency * (np.cos(frequency * times) - np.cos(omega * times)) / (omega**2 - frequency**2)


def build_photon_reference(final_time: float = 3.0, evaluation_points: int = 6001) -> PhotonReference:
    times = np.linspace(0.0, final_time, evaluation_points)
    coordinate, coordinate_velocity, coordinate_acceleration = photon_coordinate(times)

    def vector_integrand(omega_value: float) -> Array:
        kernel_values = np.cos(omega_value * times)
        convolution_values = (
            (-Q0 * Q_FREQ_0) * _cos_sin_convolution(np.asarray(omega_value), Q_FREQ_0, times)
            + (Q1 * Q_FREQ_1) * _cos_cos_convolution(np.asarray(omega_value), Q_FREQ_1, times)
        )
        return float(memory_density(omega_value)) * np.concatenate([kernel_values, convolution_values])

    continuum_values, quadrature_error = quad_vec(
        vector_integrand,
        OMEGA_MIN,
        OMEGA_MAX,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=300,
    )
    kernel = np.asarray(continuum_values[:evaluation_points], dtype=float)
    convolution = np.asarray(continuum_values[evaluation_points:], dtype=float)
    source = coordinate_acceleration + SELECTED_FREQUENCY**2 * coordinate + convolution + kernel * coordinate[0]
    return PhotonReference(
        times=times,
        coordinate=coordinate,
        coordinate_velocity=coordinate_velocity,
        coordinate_acceleration=coordinate_acceleration,
        kernel=kernel,
        convolution=convolution,
        source=source,
        quadrature_error_estimate=float(quadrature_error),
    )


def photon_rule(order: int, family: str) -> tuple[Array, Array]:
    if family == "gauss-legendre":
        nodes, weights = np.polynomial.legendre.leggauss(order)
        omega = 0.5 * (OMEGA_MAX - OMEGA_MIN) * nodes + 0.5 * (OMEGA_MAX + OMEGA_MIN)
        measure_weights = 0.5 * (OMEGA_MAX - OMEGA_MIN) * weights * memory_density(omega)
    elif family == "midpoint":
        omega = OMEGA_MIN + (np.arange(order) + 0.5) * (OMEGA_MAX - OMEGA_MIN) / order
        measure_weights = (OMEGA_MAX - OMEGA_MIN) / order * memory_density(omega)
    else:
        raise ValueError(f"unknown quadrature family: {family}")
    return np.asarray(omega, dtype=float), np.asarray(measure_weights, dtype=float)


def photon_quadrature_source(reference: PhotonReference, order: int, family: str) -> tuple[Array, Array]:
    omega, weights = photon_rule(order, family)
    omega_column = omega[:, None]
    time_row = reference.times[None, :]
    kernel = np.sum(weights[:, None] * np.cos(omega_column * time_row), axis=0)
    convolution = np.sum(
        weights[:, None]
        * (
            (-Q0 * Q_FREQ_0) * _cos_sin_convolution(omega_column, Q_FREQ_0, time_row)
            + (Q1 * Q_FREQ_1) * _cos_cos_convolution(omega_column, Q_FREQ_1, time_row)
        ),
        axis=0,
    )
    source = reference.coordinate_acceleration + SELECTED_FREQUENCY**2 * reference.coordinate + convolution + kernel * reference.coordinate[0]
    return source, kernel


def _scientific(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.{digits}f}\times10^{{{exponent}}}"


def write_tex_outputs(output_dir: Path, electron_results: list[FiniteElectronResult], photon_rows: list[dict], electron_reference: ElectronReference) -> None:
    balanced = {r.distance: r for r in electron_results if r.sequence == "balanced"}
    asymmetric = {r.distance: r for r in electron_results if r.sequence == "asymmetric"}
    row_lookup = {(row["family"], row["order"]): row for row in photon_rows}
    family_spread = {
        distance: float(np.max(np.abs(balanced[distance].source - asymmetric[distance].source)))
        for distance in balanced
    }
    macros = [
        "\\newcommand{\\ElectronErrNear}{\\ensuremath{%s}}" % _scientific(balanced[3].continuous_source_error),
        "\\newcommand{\\ElectronErrFar}{\\ensuremath{%s}}" % _scientific(balanced[9].continuous_source_error, digits=1),
        "\\newcommand{\\ElectronErrAsymFar}{\\ensuremath{%s}}" % _scientific(asymmetric[9].continuous_source_error, digits=1),
        "\\newcommand{\\ElectronFamilySpreadFar}{\\ensuremath{%s}}" % _scientific(family_spread[9], digits=1),
        "\\newcommand{\\ElectronKappaMin}{\\ensuremath{%.4f}}" % min(r.kappa_min for r in electron_results),
        "\\newcommand{\\ElectronReferenceRefinement}{\\ensuremath{%s}}" % _scientific(electron_reference.refinement_density_acceleration),
        "\\newcommand{\\ElectronReplayDensity}{\\ensuremath{%s}}" % _scientific(max(r.replay_density_error for r in electron_results)),
        "\\newcommand{\\ElectronReplayState}{\\ensuremath{%s}}" % _scientific(max(r.replay_state_error for r in electron_results)),
        "\\newcommand{\\ElectronNormDrift}{\\ensuremath{%s}}" % _scientific(max(r.norm_drift for r in electron_results)),
        "\\newcommand{\\PhotonErrLow}{\\ensuremath{%s}}" % _scientific(row_lookup[("gauss-legendre", 2)]["source_error"]),
        "\\newcommand{\\PhotonErrHigh}{\\ensuremath{%s}}" % _scientific(row_lookup[("gauss-legendre", 8)]["source_error"]),
    ]
    (output_dir / "benchmark_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    electronic_lines = [
        r"\begin{tabular}{c@{\quad}rrrr}", r"\toprule",
        r"$d$ & balanced $\epsilon_v$ & asymmetric $\epsilon_v$ & family spread & $\min_t\kappa$\\", r"\midrule",
    ]
    for distance in sorted(balanced):
        b_result, a_result = balanced[distance], asymmetric[distance]
        kappa_min = min(b_result.kappa_min, a_result.kappa_min)
        displayed_digits = 1 if distance == 9 else 2
        electronic_lines.append(
            rf"{distance} & $ {_scientific(b_result.continuous_source_error, digits=displayed_digits)} $ & $ {_scientific(a_result.continuous_source_error, digits=displayed_digits)} $ & $ {_scientific(family_spread[distance], digits=displayed_digits)} $ & {kappa_min:.4f}\\"
        )
    electronic_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "benchmark_electronic_table.tex").write_text("\n".join(electronic_lines) + "\n", encoding="utf-8")

    selected_orders = [2, 4, 6, 8, 12]
    photon_lines = [
        r"\begin{tabular}{c@{\quad}rr@{\qquad}rr}", r"\toprule",
        r"&\multicolumn{2}{c}{Gauss--Legendre}&\multicolumn{2}{c}{positive midpoint}\\",
        r"$M$ & $\epsilon_j$ & $\epsilon_\gamma$ & $\epsilon_j$ & $\epsilon_\gamma$\\", r"\midrule",
    ]
    for order in selected_orders:
        gl, mid = row_lookup[("gauss-legendre", order)], row_lookup[("midpoint", order)]
        photon_lines.append(
            rf"{order} & $ {_scientific(gl['source_error'])} $ & $ {_scientific(gl['kernel_error'])} $ & $ {_scientific(mid['source_error'])} $ & $ {_scientific(mid['kernel_error'])} $\\"
        )
    photon_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "benchmark_photon_table.tex").write_text("\n".join(photon_lines) + "\n", encoding="utf-8")


def make_figures(figure_dir: Path, electron_reference: ElectronReference, electron_results: list[FiniteElectronResult], photon_reference: PhotonReference, photon_data: dict[tuple[str, int], tuple[Array, Array]], photon_rows: list[dict]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    balanced = {r.distance: r for r in electron_results if r.sequence == "balanced"}
    asymmetric = {r.distance: r for r in electron_results if r.sequence == "asymmetric"}

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), constrained_layout=True)
    # Direct labels are used instead of legends so that no curve is hidden by a legend box.
    residual_styles = {4: "-", 6: "--", 8: "-.", 9: ":"}
    for distance in [4, 6, 8, 9]:
        result = balanced[distance]
        line, = axes[0, 0].semilogy(
            result.times,
            np.maximum(result.source_error, 1.0e-16),
            linestyle=residual_styles[distance],
        )
        axes[0, 0].annotate(
            rf"$d={distance}$",
            xy=(result.times[-1], max(result.source_error[-1], 1.0e-16)),
            xytext=(5, 0),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=7,
            va="center",
            clip_on=False,
        )
    axes[0, 0].set_xlim(0.0, 3.35)
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(r"$|v_d(t)-v_\infty(t)|$")
    axes[0, 0].set_title("(a) Infinite-lattice source residual")

    distances = sorted(balanced)
    line_bal, = axes[0, 1].semilogy(
        distances,
        [balanced[d].continuous_source_error for d in distances],
        marker="o",
    )
    line_asym, = axes[0, 1].semilogy(
        distances,
        [asymmetric[d].continuous_source_error for d in distances],
        marker="s",
        linestyle="--",
    )
    axes[0, 1].text(
        9.18, 1.2e-6, "balanced",
        color=line_bal.get_color(), fontsize=7, ha="left", va="center"
    )
    axes[0, 1].text(
        9.18, 3.2e-7, "asymmetric",
        color=line_asym.get_color(), fontsize=7, ha="left", va="center"
    )
    axes[0, 1].set_xlim(min(distances) - 0.35, max(distances) + 2.5)
    axes[0, 1].set_xlabel("nearest-boundary distance $d$")
    axes[0, 1].set_ylabel(r"$\max_t |\Delta v|$")
    axes[0, 1].set_title("(b) Two lead sequences")

    photon_styles = {2: "-", 4: "--", 6: "-.", 8: ":"}
    for order in [2, 4, 6, 8]:
        source, _ = photon_data[("gauss-legendre", order)]
        residual = np.maximum(np.abs(source - photon_reference.source), 1.0e-16)
        line, = axes[1, 0].semilogy(
            photon_reference.times, residual, linestyle=photon_styles[order]
        )
        axes[1, 0].annotate(
            rf"$M={order}$",
            xy=(photon_reference.times[-1], residual[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=7,
            va="center",
            clip_on=False,
        )
    axes[1, 0].set_xlim(0.0, 3.35)
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r"$|j_M(t)-j_\infty(t)|$")
    axes[1, 0].set_title("(c) Continuum-source residual")

    continuum_lines = {}
    for family, marker, linestyle, label in [
        ("gauss-legendre", "o", "-", "Gauss--Legendre"),
        ("midpoint", "s", "--", "positive midpoint"),
    ]:
        rows = [row for row in photon_rows if row["family"] == family and (family != "gauss-legendre" or row["order"] <= 12)]
        line, = axes[1, 1].loglog(
            [row["order"] for row in rows],
            [max(row["source_error"], 1.0e-16) for row in rows],
            marker=marker,
            linestyle=linestyle,
        )
        continuum_lines[family] = (line, rows, label)
    line_gl, _, _ = continuum_lines["gauss-legendre"]
    line_mid, _, _ = continuum_lines["midpoint"]
    axes[1, 1].text(
        13.5, 1.2e-13, "Gauss--Legendre",
        color=line_gl.get_color(), fontsize=7, ha="left", va="center"
    )
    axes[1, 1].text(
        68.0, 1.5e-6, "positive midpoint",
        color=line_mid.get_color(), fontsize=7, ha="left", va="center"
    )
    axes[1, 1].set_xlim(1.7, 125.0)
    axes[1, 1].set_xlabel("quadrature order $M$")
    axes[1, 1].set_ylabel(r"$\max_{t_k} |\Delta j|$")
    axes[1, 1].set_title("(d) Two continuum discretizations")

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
        axis.title.set_size(8)
    fig.savefig(figure_dir / "exact_reservoir_benchmark.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "exact_reservoir_benchmark.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    sample = slice(None, None, 20)
    axes[0, 0].plot(electron_reference.times[sample], electron_reference.density[sample])
    axes[0, 0].set_title("(a) Infinite-lattice device occupation")
    axes[0, 0].set_ylabel(r"$n_0(t)$")
    axes[0, 1].plot(electron_reference.times[sample], electron_reference.source[sample])
    axes[0, 1].set_title("(b) Prescribed local potential")
    axes[0, 1].set_ylabel(r"$v_\infty(t)$")
    axes[1, 0].plot(photon_reference.times, photon_reference.coordinate)
    axes[1, 0].set_title("(c) Analytic selected coordinate")
    axes[1, 0].set_ylabel(r"$Q(t)$")
    axes[1, 1].plot(photon_reference.times, photon_reference.source)
    axes[1, 1].set_title("(d) Direct continuum drive")
    axes[1, 1].set_ylabel(r"$j_\infty(t)$")
    for axis in axes.flat:
        axis.set_xlabel("time")
        axis.grid(True, alpha=0.25)
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
        axis.title.set_size(8)
    fig.savefig(figure_dir / "exact_reference_histories.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "exact_reference_histories.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_publication_figures(
    figure_dir: Path,
    electron_rows: list[dict],
    photon_rows: list[dict],
) -> None:
    """Write the two compact convergence figures included in the manuscript."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    for sequence, marker, linestyle in [
        ("balanced", "o", "-"),
        ("asymmetric", "s", "--"),
    ]:
        rows = sorted(
            (row for row in electron_rows if row["sequence"] == sequence),
            key=lambda row: int(row["distance"]),
        )
        distances = [int(row["distance"]) for row in rows]
        errors = [float(row["source_error"]) for row in rows]
        line, = axis.semilogy(distances, errors, marker=marker, linestyle=linestyle)
        offset = (7, -6) if sequence == "balanced" else (7, 6)
        axis.annotate(
            sequence,
            xy=(distances[-1], errors[-1]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.3,
            color=line.get_color(),
            ha="left",
            va="center",
            clip_on=False,
        )
    all_distances = [int(row["distance"]) for row in electron_rows]
    axis.set_xlim(min(all_distances) - 0.3, max(all_distances) + 2.2)
    axis.set_xlabel(r"nearest-boundary parameter $d$")
    axis.set_ylabel(r"$\max_t|v_d(t)-v_\infty(t)|$")
    axis.grid(True, which="both", alpha=0.25)
    axis.tick_params(labelsize=7.5)
    axis.xaxis.label.set_size(8.5)
    axis.yaxis.label.set_size(8.5)
    fig.savefig(figure_dir / "electron_exact_limit.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "electron_exact_limit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    for family, marker, linestyle, label in [
        ("gauss-legendre", "o", "-", "Gauss--Legendre"),
        ("midpoint", "s", "--", "positive midpoint"),
    ]:
        rows = sorted(
            (
                row for row in photon_rows
                if row["family"] == family and int(row["order"]) <= 12
            ),
            key=lambda row: int(row["order"]),
        )
        orders = [int(row["order"]) for row in rows]
        errors = [float(row["source_error"]) for row in rows]
        line, = axis.loglog(orders, errors, marker=marker, linestyle=linestyle)
        offset = (7, -3 if family == "gauss-legendre" else 5)
        axis.annotate(
            label,
            xy=(orders[-1], errors[-1]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.2,
            color=line.get_color(),
            ha="left",
            va="center",
            clip_on=False,
        )
    axis.set_xlim(1.75, 24)
    axis.set_xlabel(r"photon quadrature order $M$")
    axis.set_ylabel(r"$\max_t|j_M(t)-j_\infty(t)|$")
    axis.grid(True, which="both", alpha=0.25)
    axis.tick_params(labelsize=7.5)
    axis.xaxis.label.set_size(8.5)
    axis.yaxis.label.set_size(8.5)
    fig.savefig(figure_dir / "photon_exact_limit.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "photon_exact_limit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--final-time", type=float, default=3.0)
    parser.add_argument("--fine-points", type=int, default=96001)
    args = parser.parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    electron_reference = build_electron_reference(args.final_time, args.fine_points)
    electron_results: list[FiniteElectronResult] = []
    for distance in range(3, 10):
        electron_results.append(finite_electron_closed_loop(electron_reference, np.arange(-distance, distance + 1), "balanced", distance))
        electron_results.append(finite_electron_closed_loop(electron_reference, np.arange(-distance, 2 * distance + 1), "asymmetric", distance))

    photon_reference = build_photon_reference(args.final_time)
    photon_orders = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    photon_data: dict[tuple[str, int], tuple[Array, Array]] = {}
    photon_rows: list[dict] = []
    for family in ("gauss-legendre", "midpoint"):
        for order in photon_orders:
            source, kernel = photon_quadrature_source(photon_reference, order, family)
            photon_data[(family, order)] = (source, kernel)
            photon_rows.append({
                "family": family,
                "order": order,
                "source_error": float(np.max(np.abs(source - photon_reference.source))),
                "kernel_error": float(np.max(np.abs(kernel - photon_reference.kernel))),
            })

    spread_by_distance: dict[int, float] = {}
    for distance in range(3, 10):
        balanced_result = next(r for r in electron_results if r.sequence == "balanced" and r.distance == distance)
        asymmetric_result = next(r for r in electron_results if r.sequence == "asymmetric" and r.distance == distance)
        spread_by_distance[distance] = float(np.max(np.abs(balanced_result.source - asymmetric_result.source)))

    electron_rows = [{
        "sequence": result.sequence,
        "distance": result.distance,
        "length": int(len(result.coordinates)),
        "left_boundary": int(result.coordinates[0]),
        "right_boundary": int(result.coordinates[-1]),
        "source_error": result.continuous_source_error,
        "source_error_grid": result.grid_source_error,
        "source_error_argmax": result.continuous_argmax,
        "density_error_against_infinite_target": float(np.max(result.density_error)),
        "minimum_kappa_dense_interpolant": result.kappa_min,
        "minimum_kappa_argmin": result.kappa_argmin,
        "minimum_kappa_reporting_grid": result.kappa_grid_min,
        "closed_loop_norm_drift": result.norm_drift,
        "forward_replay_density_error": result.replay_density_error,
        "forward_replay_state_difference": result.replay_state_error,
        "forward_replay_norm_drift": result.replay_norm_drift,
        "family_source_spread": spread_by_distance[result.distance],
    } for result in electron_results]

    with (args.output_dir / "electronic_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(electron_rows[0]))
        writer.writeheader(); writer.writerows(electron_rows)
    with (args.output_dir / "photon_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(photon_rows[0]))
        writer.writeheader(); writer.writerows(photon_rows)

    balanced_lookup = {r.distance: r for r in electron_results if r.sequence == "balanced"}
    trace_rows = []
    for index, time_value in enumerate(balanced_lookup[4].times):
        row = {"time": float(time_value)}
        for distance in [4, 6, 8, 9]:
            row[f"electron_abs_error_d{distance}"] = float(balanced_lookup[distance].source_error[index])
        for order in [2, 4, 6, 8]:
            photon_residual = np.abs(
                photon_data[("gauss-legendre", order)][0] - photon_reference.source
            )
            row[f"photon_abs_error_M{order}"] = float(
                np.interp(time_value, photon_reference.times, photon_residual)
            )
        trace_rows.append(row)
    with (args.output_dir / "benchmark_residual_traces.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader(); writer.writerows(trace_rows)

    result_json = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "elapsed_seconds": time.perf_counter() - start,
        },
        "electronic_reference": {
            "final_time": args.final_time,
            "fine_points": args.fine_points,
            "source_definition": "0.18 sin(0.73 t) + 0.055 cos(1.31 t)",
            "finite_inverse_workflow": "closed-loop stage-wise force-balance reconstruction; exact source used only for target manufacture and scoring",
            "initial_sites": INITIAL_SITES.tolist(),
            "initial_amplitudes": INITIAL_AMPLITUDES.tolist(),
            "refinement_psi0": electron_reference.refinement_psi0,
            "refinement_density_acceleration": electron_reference.refinement_density_acceleration,
            "minimum_exact_kappa": float(np.min(electron_reference.kappa)),
        },
        "electronic_convergence": electron_rows,
        "photon_reference": {
            "frequency_interval": [OMEGA_MIN, OMEGA_MAX],
            "memory_mass": MEMORY_MASS,
            "memory_center": MEMORY_CENTER,
            "memory_width": MEMORY_WIDTH,
            "selected_frequency": SELECTED_FREQUENCY,
            "coordinate_definition": "0.15 cos(0.42 t) + 0.07 sin(2.20 t)",
            "adaptive_quadrature_error_estimate": photon_reference.quadrature_error_estimate,
            "memory_mass_check": float(quad(lambda value: float(memory_density(value)), OMEGA_MIN, OMEGA_MAX, epsabs=1e-14)[0]),
        },
        "photon_convergence": photon_rows,
    }
    (args.output_dir / "exact_reservoir_benchmark.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")

    write_tex_outputs(args.output_dir, electron_results, photon_rows, electron_reference)
    make_publication_figures(args.figure_dir, electron_rows, photon_rows)

    print(json.dumps({
        "elapsed_seconds": result_json["metadata"]["elapsed_seconds"],
        "electronic_near_error": next(row["source_error"] for row in electron_rows if row["sequence"] == "balanced" and row["distance"] == 3),
        "electronic_far_error": next(row["source_error"] for row in electron_rows if row["sequence"] == "balanced" and row["distance"] == 9),
        "minimum_kappa": min(row["minimum_kappa_dense_interpolant"] for row in electron_rows),
        "photon_M2_error": next(row["source_error"] for row in photon_rows if row["family"] == "gauss-legendre" and row["order"] == 2),
        "photon_M8_error": next(row["source_error"] for row in photon_rows if row["family"] == "gauss-legendre" and row["order"] == 8),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
