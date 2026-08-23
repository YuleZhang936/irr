# RTQEDFT numerical reproduction

Minimal code for reproducing every numerical value in the numerical-benchmark
section and all three panels of Fig. 2 in **“Real-Time Density–Source Mapping
for Open Electron–Photon Systems.”**

The repository intentionally contains four Python files:

| File | Manuscript output |
|---|---|
| `code/benchmark_electronic.py` | SM Table S1 and Fig. 2(a) data |
| `code/benchmark_photon.py` | SM Table S2 and Fig. 2(b) data |
| `code/benchmark_joint.py` | SM Tables S3–S4, joint diagnostics, and Fig. 2(c) data |
| `code/plot_figure2.py` | The three Fig. 2 panels from the generated CSV files |

## Environment

Python 3.9 or newer is sufficient.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce

```bash
make reproduce PYTHON=.venv/bin/python
```

The electronic and photon benchmarks can be run separately with
`make electronic` and `make photon`. The coupled calculation, `make joint`, performs
36 production cases plus the cutoff and tolerance repeats used to certify the
reported mixed-regulator defect; it is the expensive part of the reproduction.

Generated CSV files are written to `results/`. Figure 2 panel files are written
as PDF and PNG to `figures/`.

The primary outputs are:

```text
results/electronic_convergence.csv
results/photon_convergence.csv
results/joint_primary.csv
results/joint_family_summary.csv
results/joint_diagnostics.csv
results/joint_all_cases.csv
results/joint_analytic_inverse_cross.csv
figures/electron_exact_limit.pdf
figures/photon_exact_limit.pdf
figures/joint_analytic_inverse.pdf
```
