

## Environment and commands


```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5 scipy==1.17.0 matplotlib==3.10.8
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

python test_1_constraint.py
python test_2_ground_state.py
python test_3_real_time.py
```


## Files generated

| Script | Data | Figure files |
|---|---|---|
| `test_1_constraint.py` | `results/test_1/ground_state_scaling.csv`, `charge_scaling.csv`, `fitted_exponents.csv` | `scaling_functional.pdf`, `scaling_potential.pdf`, `scaling_charge.pdf`, `scaling_legend.pdf` |
| `test_2_ground_state.py` | `results/test_2/polarization_reversal.csv` | `polarization_reversal.pdf` |
| `test_3_real_time.py` | `results/test_3/spectrum.csv`, `time_trace.csv` | `tqd_spectrum.pdf`, `tqd_time.pdf`, `tqd_legend.pdf` |

All files in a row are written inside the indicated script directory. The eight PDF names match the manuscript's figure assets.


