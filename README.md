# RTQEDFT numerical benchmarks

Reproducibility code for the numerical results and figures accompanying
**“Real-Time Density-Source Mapping for Open Electron-Photon Systems.”**

本仓库只包含生成论文数值数据、表格和图片所需的代码，不包含论文源文件或预生成结果。

## Benchmarks

The repository reproduces three benchmark families:

1. **Electronic regulator.** A finite electronic chain reconstructs the local
   potential of a known infinite-lattice target. Balanced and asymmetric lead
   sequences are compared for nearest-boundary distances `d = 3, ..., 9`.
2. **Photon regulator.** Gauss–Legendre and positive-midpoint quadratures
   approximate a compact, gapped photon continuum for several orders `M`.
3. **Joint inverse.** A coupled electron–photon model is evaluated on
   `L = (4, 8, 12)` and `M = (1, 2, 4)`, using two electronic termination
   families and two photon quadrature families (36 production cases).

The main reported errors are

- `epsilon_v = max_t |v_L(t) - v_infinity(t)|`;
- `epsilon_j = max_t |j_M(t) - j_infinity(t)|`;
- `epsilon_gamma`, the photon-memory-kernel error;
- `E_src = max(epsilon_v, epsilon_j)` for the joint benchmark.

Cutoff repeats, tighter-tolerance repeats, density/coordinate tracking, norm
drift, response-gap checks, and the four-corner mixed defect are also recorded.

## Code map

| File | Purpose | Main outputs |
|---|---|---|
| `code/separated.py` | Compute the electronic and photon benchmarks | CSV, JSON, TeX tables/macros, diagnostic traces |
| `code/plot_separated.py` | Plot the two separate manuscript figures | `electron_exact_limit.*`, `photon_exact_limit.*` |
| `code/joint.py` | Propagate one or many coupled `(L, M)` cases | compressed raw-case NPZ files |
| `code/run_joint.sh` | Run all production cases and numerical repeats | `results/joint/raw_cases/`, `results/joint/checks/` |
| `code/aggregate.py` | Aggregate joint cases into publication artifacts | CSV, JSON, TeX table/macros, heatmap |
| `code/schematic.py` | Draw the model schematic | `device_schematic.pdf/png` |

## Requirements

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` gives compatible version ranges. `requirements-lock.txt`
records the exact package versions used for the archived production run.

## Reproduce everything

```bash
make reproduce
```

This command removes old generated artifacts and recreates all numerical data,
tables, and figures. The joint sparse propagation is the expensive stage.

Individual targets are available:

```bash
make separated  # electronic + photon data and figures
make joint      # coupled (L, M) data, checks, table, and heatmap
make schematic  # device schematic only
```

CPU math-library thread counts are fixed to one by the `Makefile` for more
predictable runs.

## Generated layout

```text
results/
├── first_two/
│   ├── electronic_convergence.csv
│   ├── photon_convergence.csv
│   ├── benchmark_residual_traces.csv
│   ├── exact_reservoir_benchmark.json
│   └── *.tex
└── joint/
    ├── raw_cases/*.npz
    ├── checks/*.npz
    └── summary/
        ├── joint_analytic_inverse_grid.csv
        ├── joint_analytic_inverse_cross.csv
        ├── joint_analytic_inverse_benchmark.json
        └── *.tex

figures/
├── electron_exact_limit.pdf/png
├── photon_exact_limit.pdf/png
├── joint_analytic_inverse.pdf/png
└── device_schematic.pdf/png
```

Generated `results/`, `figures/`, and local environment/cache directories are
ignored by Git.
