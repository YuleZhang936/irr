#!/usr/bin/env python3
"""Reproduce the electronic-regulator data in SM Table S1 / Fig. 2(a).

Run
    python benchmark_electronic.py --output-dir results

Output
    results/electronic_convergence.csv

Dependencies: numpy, scipy.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.special import jv

HOPPING = 1.0
INITIAL_SITES = np.arange(-2, 3)
_INITIAL_AMPLITUDES = np.array([0.20, 0.45, 0.70, 0.45, 0.20], dtype=float)
INITIAL_AMPLITUDES = _INITIAL_AMPLITUDES / np.linalg.norm(_INITIAL_AMPLITUDES)


def electronic_source(t):
    t = np.asarray(t)
    return 0.18 * np.sin(0.73 * t) + 0.055 * np.cos(1.31 * t)


def free_lattice_amplitude(site: int, times: np.ndarray) -> np.ndarray:
    out = np.zeros_like(times, dtype=np.complex128)
    for initial_site, amplitude in zip(INITIAL_SITES, INITIAL_AMPLITUDES):
        order = site - int(initial_site)
        out += (1j ** order) * jv(order, 2.0 * HOPPING * times) * amplitude
    return out


def build_infinite_target(final_time: float = 3.0, time_points: int = 96001):
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
        rhs = free0[index] - 1j * step * (0.5 * kernel0[index] * forcing[0] + interior)
        psi0[index] = rhs / (1.0 + 0.5j * step * source[index])
        forcing[index] = source[index] * psi0[index]

    amplitudes = {0: psi0}
    for site in (-2, -1, 1, 2):
        free_kernel = (1j ** site) * jv(site, 2.0 * HOPPING * times)
        discrete_convolution = fftconvolve(free_kernel, forcing)[:time_points]
        integral = step * (
            discrete_convolution
            - 0.5 * free_kernel * forcing[0]
            - 0.5 * free_kernel[0] * forcing
        )
        amplitudes[site] = free_lattice_amplitude(site, times) - 1j * integral

    hpsi0 = -HOPPING * (amplitudes[-1] + amplitudes[1]) + source * amplitudes[0]
    hpsim1 = -HOPPING * (amplitudes[-2] + amplitudes[0])
    hpsip1 = -HOPPING * (amplitudes[0] + amplitudes[2])
    h2psi0 = -HOPPING * (hpsim1 + hpsip1) + source * hpsi0
    density_acceleration = (
        2.0 * np.abs(hpsi0) ** 2
        - 2.0 * np.real(np.conj(amplitudes[0]) * h2psi0)
    )
    density = np.abs(amplitudes[0]) ** 2
    return times, density, density_acceleration


def finite_source_error(
    target_times: np.ndarray,
    target_density: np.ndarray,
    target_acceleration: np.ndarray,
    coordinates: np.ndarray,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-14,
) -> tuple[float, float]:
    coordinates = np.asarray(coordinates, dtype=int)
    site_to_index = {int(site): index for index, site in enumerate(coordinates)}
    device_index = site_to_index[0]
    length = len(coordinates)

    h0 = np.zeros((length, length), dtype=np.complex128)
    for site in coordinates:
        if int(site) + 1 in site_to_index:
            i = site_to_index[int(site)]
            j = site_to_index[int(site) + 1]
            h0[i, j] = h0[j, i] = -HOPPING

    initial_state = np.zeros(length, dtype=np.complex128)
    for site, amplitude in zip(INITIAL_SITES, INITIAL_AMPLITUDES):
        initial_state[site_to_index[int(site)]] = amplitude

    target_ndd = CubicSpline(target_times, target_acceleration)

    def reconstruct(time_value: float, state: np.ndarray) -> tuple[float, float]:
        norm2 = float(np.vdot(state, state).real)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            h0_state = h0 @ state
            h0_squared_state = h0 @ h0_state
        first = -1j * h0_state
        second = -h0_squared_state
        internal = float(
            (
                2.0 * abs(first[device_index]) ** 2
                + 2.0 * np.real(np.conj(state[device_index]) * second[device_index])
            )
            / norm2
        )
        k_value = 0.0
        for neighbour in (-1, 1):
            if neighbour in site_to_index:
                ni = site_to_index[neighbour]
                k_value += 2.0 * np.real(
                    h0[device_index, ni] * np.conj(state[device_index]) * state[ni]
                ) / norm2
        if abs(k_value) < 1.0e-9:
            raise RuntimeError(f"device matrix became singular at t={time_value}: K={k_value}")
        potential = (float(target_ndd(time_value)) - internal) / k_value
        return float(potential), float(-k_value)

    def rhs(time_value: float, state: np.ndarray) -> np.ndarray:
        potential, _ = reconstruct(time_value, state)
        derivative = h0 @ state
        derivative[device_index] += potential * state[device_index]
        return -1j * derivative

    final_time = float(target_times[-1])
    # DOP853 can evaluate rejected trial stages with very large amplitudes in
    # the coarsest finite volumes. The accepted solution is checked below.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
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
        raise RuntimeError(solution.message)

    def residual(time_value: float) -> float:
        state = solution.sol(float(time_value))
        potential, _ = reconstruct(float(time_value), state)
        return abs(potential - float(electronic_source(float(time_value))))

    dense_times = np.linspace(0.0, final_time, 6001)
    dense_values = np.asarray([residual(float(t)) for t in dense_times])
    candidates = {0, len(dense_times) - 1, int(np.argmax(dense_values))}
    for i in range(1, len(dense_times) - 1):
        if dense_values[i] >= dense_values[i - 1] and dense_values[i] >= dense_values[i + 1]:
            candidates.add(i)
    best_value = float(np.max(dense_values))
    best_time = float(dense_times[int(np.argmax(dense_values))])
    dt = float(dense_times[1] - dense_times[0])
    for i in sorted(candidates):
        lo = max(0.0, float(dense_times[i] - 2.0 * dt))
        hi = min(final_time, float(dense_times[i] + 2.0 * dt))
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
    return best_value, best_time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--final-time", type=float, default=3.0)
    parser.add_argument("--target-points", type=int, default=96001)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    times, density, acceleration = build_infinite_target(args.final_time, args.target_points)
    rows = []
    for distance in range(3, 10):
        for sequence, coordinates in (
            ("balanced", np.arange(-distance, distance + 1)),
            ("asymmetric", np.arange(-distance, 2 * distance + 1)),
        ):
            error, argmax = finite_source_error(times, density, acceleration, coordinates)
            rows.append(
                {
                    "sequence": sequence,
                    "distance": distance,
                    "source_error": error,
                    "argmax_time": argmax,
                }
            )
            print(f"{sequence:10s} d={distance}: epsilon_v={error:.12e}")

    out = args.output_dir / "electronic_convergence.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "distance", "source_error", "argmax_time"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
