.PHONY: all diva-tag-softmax diva-tag-softplus diva-gather-softmax diva-gather-softplus diva-tag-runs diva-gather-runs diva-all-runs
.PHONY: diva-theorem-gather diva-theorem-hallway diva-theorem-disperse diva-theorem-pursuit diva-theorem-runs
.PHONY: qmix-gather qmix-hallway qmix-disperse qmix-runs
.PHONY: qmix-top5-gather qmix-top5-hallway qmix-top5-disperse qmix-top5-runs
.PHONY: vdn-top5-gather vdn-top5-hallway vdn-top5-disperse vdn-top5-runs
.PHONY: qtran-top5-gather qtran-top5-hallway qtran-top5-disperse qtran-top5-runs
.PHONY: baseline-top5-gather baseline-top5-hallway baseline-top5-disperse baseline-top5-runs
.PHONY: pursuit-then-baseline-top5-runs

SHELL := /bin/bash
PYTHON ?= python
USE_CUDA ?= True
DIVA_THEOREM_CONFIG ?= diva_bounded_sigmoid_qmix_DIVA
QMIX_CONFIG ?= qmix
VDN_CONFIG ?= vdn
QTRAN_CONFIG ?= qtran
GPU_COUNT ?= 4
MAX_JOBS_PER_GPU ?= 2

PURSUIT_SEEDS ?= 1 2 3 4 5 6 7 8
GATHER_TOP5_SEEDS ?= 2 4 3 1 8
HALLWAY_TOP5_SEEDS ?= 1 5 3 8 2
DISPERSE_TOP5_SEEDS ?= 8 5 3 1 7

define run_seed_waves
set -eu; \
config="$(1)"; \
env_cfg="$(2)"; \
seeds="$(3)"; \
max_parallel=$$(( $(GPU_COUNT) * $(MAX_JOBS_PER_GPU) )); \
set -- $$seeds; \
wave=1; \
while [ $$# -gt 0 ]; do \
	echo "[config=$$config env=$$env_cfg] wave $$wave"; \
	slot=0; \
	while [ $$slot -lt $$max_parallel ] && [ $$# -gt 0 ]; do \
		seed="$$1"; \
		shift; \
		gpu=$$((slot / $(MAX_JOBS_PER_GPU))); \
		echo "  launching seed=$$seed on gpu=$$gpu"; \
		CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_cfg with seed=$$seed use_cuda=$(USE_CUDA) & \
		slot=$$((slot + 1)); \
	done; \
	wait; \
	wave=$$((wave + 1)); \
done
endef

all:
	clear
	# Priority order:
	# 1. tag / softmax_DIVA
	# 2. tag / softplus_DIVA
	# 3. gather / softmax_DIVA
	# 4. gather / softplus_DIVA
	#
	# Each target launches exactly 8 jobs:
	# 2 jobs on each of 4 GPUs, using seeds 1..8.
	# The recipe waits for those 8 jobs to finish before returning.
	#
	# Example:
	# make diva-tag-softmax PYTHON=/home/ludwika/miniconda3/envs/gacg/bin/python
	# make diva-tag-runs PYTHON=/home/ludwika/miniconda3/envs/gacg/bin/python
	# make diva-all-runs PYTHON=/home/ludwika/miniconda3/envs/gacg/bin/python

diva-tag-softmax:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=8 use_cuda=$(USE_CUDA) & \
	wait

diva-tag-softplus:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=8 use_cuda=$(USE_CUDA) & \
	wait

diva-gather-softmax:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=8 use_cuda=$(USE_CUDA) & \
	wait

diva-gather-softplus:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=8 use_cuda=$(USE_CUDA) & \
	wait

diva-tag-runs:
	$(MAKE) diva-tag-softmax PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)
	$(MAKE) diva-tag-softplus PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)

diva-gather-runs:
	$(MAKE) diva-gather-softmax PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)
	$(MAKE) diva-gather-softplus PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)

diva-all-runs:
	$(MAKE) diva-tag-runs PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)
	$(MAKE) diva-gather-runs PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)

# Theorem-oriented DIVA sweeps:
# - uses `diva_bounded_sigmoid_qmix_DIVA`
# - launches 8 seeds across 4 GPUs
# - runs each environment sweep sequentially when chained via `diva-theorem-runs`

diva-theorem-gather:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=gather with seed=8 use_cuda=$(USE_CUDA) & \
	wait

diva-theorem-hallway:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=hallway with seed=8 use_cuda=$(USE_CUDA) & \
	wait

diva-theorem-disperse:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(DIVA_THEOREM_CONFIG) --env-config=disperse with seed=8 use_cuda=$(USE_CUDA) & \
	wait

diva-theorem-pursuit:
	@$(call run_seed_waves,$(DIVA_THEOREM_CONFIG),pursuit,$(PURSUIT_SEEDS))

diva-theorem-runs:
	$(MAKE) diva-theorem-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) DIVA_THEOREM_CONFIG=$(DIVA_THEOREM_CONFIG)
	$(MAKE) diva-theorem-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) DIVA_THEOREM_CONFIG=$(DIVA_THEOREM_CONFIG)
	$(MAKE) diva-theorem-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) DIVA_THEOREM_CONFIG=$(DIVA_THEOREM_CONFIG)

# QMIX sweeps:
# - uses `qmix`
# - launches 8 seeds across 4 GPUs

qmix-gather:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=gather with seed=8 use_cuda=$(USE_CUDA) & \
	wait

qmix-hallway:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=hallway with seed=8 use_cuda=$(USE_CUDA) & \
	wait

qmix-disperse:
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=1 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=2 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=3 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=4 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=5 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=6 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=7 use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=$(QMIX_CONFIG) --env-config=disperse with seed=8 use_cuda=$(USE_CUDA) & \
	wait

qmix-runs:
	$(MAKE) qmix-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)
	$(MAKE) qmix-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)
	$(MAKE) qmix-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)

# Top-5 seeds per environment for the current DIVA runs:
# - gather:   2 4 3 1 8
# - hallway:  1 5 3 8 2
# - disperse: 8 5 3 1 7

qmix-top5-gather:
	@$(call run_seed_waves,$(QMIX_CONFIG),gather,$(GATHER_TOP5_SEEDS))

qmix-top5-hallway:
	@$(call run_seed_waves,$(QMIX_CONFIG),hallway,$(HALLWAY_TOP5_SEEDS))

qmix-top5-disperse:
	@$(call run_seed_waves,$(QMIX_CONFIG),disperse,$(DISPERSE_TOP5_SEEDS))

qmix-top5-runs:
	$(MAKE) qmix-top5-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)
	$(MAKE) qmix-top5-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)
	$(MAKE) qmix-top5-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)

vdn-top5-gather:
	@$(call run_seed_waves,$(VDN_CONFIG),gather,$(GATHER_TOP5_SEEDS))

vdn-top5-hallway:
	@$(call run_seed_waves,$(VDN_CONFIG),hallway,$(HALLWAY_TOP5_SEEDS))

vdn-top5-disperse:
	@$(call run_seed_waves,$(VDN_CONFIG),disperse,$(DISPERSE_TOP5_SEEDS))

vdn-top5-runs:
	$(MAKE) vdn-top5-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) VDN_CONFIG=$(VDN_CONFIG)
	$(MAKE) vdn-top5-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) VDN_CONFIG=$(VDN_CONFIG)
	$(MAKE) vdn-top5-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) VDN_CONFIG=$(VDN_CONFIG)

qtran-top5-gather:
	@$(call run_seed_waves,$(QTRAN_CONFIG),gather,$(GATHER_TOP5_SEEDS))

qtran-top5-hallway:
	@$(call run_seed_waves,$(QTRAN_CONFIG),hallway,$(HALLWAY_TOP5_SEEDS))

qtran-top5-disperse:
	@$(call run_seed_waves,$(QTRAN_CONFIG),disperse,$(DISPERSE_TOP5_SEEDS))

qtran-top5-runs:
	$(MAKE) qtran-top5-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QTRAN_CONFIG=$(QTRAN_CONFIG)
	$(MAKE) qtran-top5-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QTRAN_CONFIG=$(QTRAN_CONFIG)
	$(MAKE) qtran-top5-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QTRAN_CONFIG=$(QTRAN_CONFIG)

baseline-top5-gather:
	$(MAKE) qmix-top5-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)
	$(MAKE) vdn-top5-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) VDN_CONFIG=$(VDN_CONFIG)
	$(MAKE) qtran-top5-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QTRAN_CONFIG=$(QTRAN_CONFIG)

baseline-top5-hallway:
	$(MAKE) qmix-top5-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)
	$(MAKE) vdn-top5-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) VDN_CONFIG=$(VDN_CONFIG)
	$(MAKE) qtran-top5-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QTRAN_CONFIG=$(QTRAN_CONFIG)

baseline-top5-disperse:
	$(MAKE) qmix-top5-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG)
	$(MAKE) vdn-top5-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) VDN_CONFIG=$(VDN_CONFIG)
	$(MAKE) qtran-top5-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QTRAN_CONFIG=$(QTRAN_CONFIG)

baseline-top5-runs:
	$(MAKE) baseline-top5-gather PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG) VDN_CONFIG=$(VDN_CONFIG) QTRAN_CONFIG=$(QTRAN_CONFIG)
	$(MAKE) baseline-top5-hallway PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG) VDN_CONFIG=$(VDN_CONFIG) QTRAN_CONFIG=$(QTRAN_CONFIG)
	$(MAKE) baseline-top5-disperse PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG) VDN_CONFIG=$(VDN_CONFIG) QTRAN_CONFIG=$(QTRAN_CONFIG)

pursuit-then-baseline-top5-runs:
	$(MAKE) diva-theorem-pursuit PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) DIVA_THEOREM_CONFIG=$(DIVA_THEOREM_CONFIG)
	$(MAKE) baseline-top5-runs PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG) VDN_CONFIG=$(VDN_CONFIG) QTRAN_CONFIG=$(QTRAN_CONFIG)

run_baselines: 
	for f in runs_baselines/*.job; do sbatch $$f; done