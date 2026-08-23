#!/usr/bin/env python3
"""Reproduce the photon-continuum data in SM Table S2 / Fig. 2(b).

Run
    python benchmark_photon.py --output-dir results

Output
    results/photon_convergence.csv

Dependencies: numpy, scipy.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import erf

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
ORDERS = (2, 3, 4, 6, 8, 12)
FAMILIES = ("gauss-legendre", "midpoint")


def memory_density(omega):
    normalization = MEMORY_WIDTH * math.sqrt(math.pi / 2.0) * (
        erf((OMEGA_MAX - MEMORY_CENTER) / (math.sqrt(2.0) * MEMORY_WIDTH))
        - erf((OMEGA_MIN - MEMORY_CENTER) / (math.sqrt(2.0) * MEMORY_WIDTH))
    )
    return MEMORY_MASS * np.exp(
        -0.5 * ((np.asarray(omega) - MEMORY_CENTER) / MEMORY_WIDTH) ** 2
    ) / normalization


def coordinate(times: np.ndarray):
    q = Q0 * np.cos(Q_FREQ_0 * times) + Q1 * np.sin(Q_FREQ_1 * times)
    qd = -Q0 * Q_FREQ_0 * np.sin(Q_FREQ_0 * times) + Q1 * Q_FREQ_1 * np.cos(Q_FREQ_1 * times)
    qdd = -Q0 * Q_FREQ_0**2 * np.cos(Q_FREQ_0 * times) - Q1 * Q_FREQ_1**2 * np.sin(Q_FREQ_1 * times)
    return q, qd, qdd


def cos_cos_convolution(omega, frequency: float, times: np.ndarray):
    return (omega * np.sin(omega * times) - frequency * np.sin(frequency * times)) / (omega**2 - frequency**2)


def cos_sin_convolution(omega, frequency: float, times: np.ndarray):
    return frequency * (np.cos(frequency * times) - np.cos(omega * times)) / (omega**2 - frequency**2)


def continuum_reference(times: np.ndarray):
    q, _, qdd = coordinate(times)
    n = len(times)

    def integrand(omega_value: float):
        kernel = np.cos(omega_value * times)
        convolution = (
            (-Q0 * Q_FREQ_0) * cos_sin_convolution(np.asarray(omega_value), Q_FREQ_0, times)
            + (Q1 * Q_FREQ_1) * cos_cos_convolution(np.asarray(omega_value), Q_FREQ_1, times)
        )
        return float(memory_density(omega_value)) * np.concatenate([kernel, convolution])

    values, _ = quad_vec(
        integrand,
        OMEGA_MIN,
        OMEGA_MAX,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=300,
    )
    kernel = np.asarray(values[:n], dtype=float)
    convolution = np.asarray(values[n:], dtype=float)
    source = qdd + SELECTED_FREQUENCY**2 * q + convolution + kernel * q[0]
    return q, qdd, kernel, source


def quadrature_rule(order: int, family: str):
    if family == "gauss-legendre":
        nodes, weights = np.polynomial.legendre.leggauss(order)
        omega = 0.5 * (OMEGA_MAX - OMEGA_MIN) * nodes + 0.5 * (OMEGA_MAX + OMEGA_MIN)
        measure_weights = 0.5 * (OMEGA_MAX - OMEGA_MIN) * weights * memory_density(omega)
    elif family == "midpoint":
        omega = OMEGA_MIN + (np.arange(order) + 0.5) * (OMEGA_MAX - OMEGA_MIN) / order
        measure_weights = (OMEGA_MAX - OMEGA_MIN) / order * memory_density(omega)
    else:
        raise ValueError(family)
    return np.asarray(omega, dtype=float), np.asarray(measure_weights, dtype=float)


def finite_source(times: np.ndarray, q: np.ndarray, qdd: np.ndarray, order: int, family: str):
    omega, weights = quadrature_rule(order, family)
    omega_column = omega[:, None]
    time_row = times[None, :]
    kernel = np.sum(weights[:, None] * np.cos(omega_column * time_row), axis=0)
    convolution = np.sum(
        weights[:, None]
        * (
            (-Q0 * Q_FREQ_0) * cos_sin_convolution(omega_column, Q_FREQ_0, time_row)
            + (Q1 * Q_FREQ_1) * cos_cos_convolution(omega_column, Q_FREQ_1, time_row)
        ),
        axis=0,
    )
    source = qdd + SELECTED_FREQUENCY**2 * q + convolution + kernel * q[0]
    return source, kernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--final-time", type=float, default=3.0)
    parser.add_argument("--time-points", type=int, default=6001)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, args.final_time, args.time_points)
    q, qdd, kernel_inf, source_inf = continuum_reference(times)
    rows = []
    for family in FAMILIES:
        for order in ORDERS:
            source, kernel = finite_source(times, q, qdd, order, family)
            source_error = float(np.max(np.abs(source - source_inf)))
            kernel_error = float(np.max(np.abs(kernel - kernel_inf)))
            rows.append(
                {
                    "family": family,
                    "order": order,
                    "source_error": source_error,
                    "kernel_error": kernel_error,
                }
            )
            print(f"{family:16s} M={order:2d}: epsilon_j={source_error:.12e}, epsilon_gamma={kernel_error:.12e}")

    out = args.output_dir / "photon_convergence.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "order", "source_error", "kernel_error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
