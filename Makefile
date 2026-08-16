PYTHON ?= python3
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1

.PHONY: separated joint schematic reproduce clean-generated

# Electronic- and photon-regulator data, tables, and manuscript figures.
separated:
	mkdir -p $(MPLCONFIGDIR) results/first_two figures
	$(PYTHON) code/separated.py \
		--output-dir results/first_two \
		--figure-dir figures \
		--final-time 3.0 \
		--fine-points 96001
# Coupled simultaneous (L,M) inverse data and heatmap.
joint:
	mkdir -p $(MPLCONFIGDIR) results/joint/summary results/joint/figures figures
	$(PYTHON) code/joint.py \
		--output-dir results/joint/summary \
		--figure-dir results/joint/figures
	cp results/joint/figures/joint_analytic_inverse.pdf figures/
	cp results/joint/figures/joint_analytic_inverse.png figures/

# Device schematic used by the manuscript.
schematic:
	mkdir -p $(MPLCONFIGDIR) figures
	$(PYTHON) code/schematic.py

reproduce: clean-generated separated joint schematic

clean-generated:
	rm -rf results figures .mplconfig
