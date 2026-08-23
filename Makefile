PYTHON ?= python3

export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1

.PHONY: electronic photon joint data figure2 reproduce clean-generated

electronic:
	mkdir -p results
	$(PYTHON) code/benchmark_electronic.py --output-dir results

photon:
	mkdir -p results
	$(PYTHON) code/benchmark_photon.py --output-dir results

joint:
	mkdir -p results
	$(PYTHON) code/benchmark_joint.py --output-dir results

data: electronic photon joint

figure2:
	mkdir -p $(MPLCONFIGDIR) figures
	$(PYTHON) code/plot_figure2.py --data-dir results --figure-dir figures

reproduce:
	$(MAKE) clean-generated
	$(MAKE) data PYTHON=$(PYTHON)
	$(MAKE) figure2 PYTHON=$(PYTHON)

clean-generated:
	rm -rf results figures .mplconfig
