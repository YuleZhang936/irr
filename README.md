# RTQEDFT numerical benchmarks

This repository contains the numerical benchmark and figure-generation code for
**“Real-Time Density-Source Mapping for Open Electron-Photon Systems.”** It does
not contain the manuscript source or pre-generated numerical results.

## Why there are three code files

The repository intentionally contains only three Python files:

| File | Corresponding content | Generated artifacts |
|---|---|---|
| `code/separated.py` | Benchmark I (electronic regulator) and Benchmark II (photon regulator) | Electronic/photon CSV and JSON data, TeX tables and macros, and the two convergence figures |
| `code/joint.py` | Benchmark III (joint electronic-photon inverse) | Joint grid and mixed-defect data, numerical checks, TeX output, and the joint heatmap |
| `code/schematic.py` | Conceptual model figure; no numerical benchmark data | Device schematic in PDF and PNG formats |

The first two benchmarks share one file because they use the same output and
publication pipeline. The joint calculation is self-contained: case execution,
aggregation, checks, table generation, and plotting are all implemented in
`joint.py`.

## Scientific content

### I. Electronic regulator

A finite electronic chain reconstructs the local potential of a known
infinite-lattice target. Balanced and asymmetric lead sequences are evaluated
for nearest-boundary distances `d = 3, ..., 9`. The primary error is

```text
epsilon_v = max_t |v_d(t) - v_infinity(t)|.
```

The output also records the family spread, minimum inverse-response gap,
density-tracking error, norm drift, and frozen-source replay error.

### II. Photon regulator

Gauss-Legendre and positive-midpoint quadratures approximate a compact, gapped
photon continuum at increasing quadrature order `M`. The reported quantities are

```text
epsilon_j     = max_t |j_M(t) - j_infinity(t)|,
epsilon_gamma = max_t |K_M(t) - K_infinity(t)|.
```

### III. Joint electronic-photon inverse

The coupled benchmark uses

```text
L = (4, 8, 12),
M = (1, 2, 4),
```

with two electronic termination families and two photon quadrature families,
for 36 production cases. Its main error is

```text
E_src(L, M) = max(E_v(L, M), E_j(M)).
```

The same run performs four-corner cutoff and tighter-tolerance repeats and
records density/coordinate tracking, norm drift, the response gap, Fock-sector
population, regulator-family spreads, and the mixed `L-M` defect.

## Requirements

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` specifies compatible version ranges.
`requirements-lock.txt` records the exact numerical package versions used for
the archived production run.

## Reproduce all data and figures

```bash
make reproduce
```

The command removes previous generated artifacts and then runs all three code
files. The joint sparse propagation is the most expensive stage.

Individual targets are also available:

```bash
make separated  # Benchmarks I and II, including their figures
make joint      # Benchmark III, including checks, table, and heatmap
make schematic  # Conceptual device figure only
```

The `Makefile` fixes CPU math-library thread counts to one for more predictable
runs.

## Generated files

```text
results/
├── first_two/
│   ├── electronic_convergence.csv
│   ├── photon_convergence.csv
│   ├── benchmark_residual_traces.csv
│   ├── exact_reservoir_benchmark.json
│   ├── benchmark_electronic_table.tex
│   ├── benchmark_photon_table.tex
│   └── benchmark_macros.tex
└── joint/
    └── summary/
        ├── joint_analytic_inverse_grid.csv
        ├── joint_analytic_inverse_cross.csv
        ├── joint_analytic_inverse_benchmark.json
        ├── joint_analytic_inverse_table.tex
        └── joint_analytic_inverse_macros.tex

figures/
├── electron_exact_limit.pdf/png
├── photon_exact_limit.pdf/png
├── joint_analytic_inverse.pdf/png
└── device_schematic.pdf/png
```

Generated results, figures, virtual environments, and local cache directories
are excluded by `.gitignore`.
