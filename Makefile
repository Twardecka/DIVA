.PHONY: all diva-tag-softmax diva-tag-softplus diva-gather-softmax diva-gather-softplus diva-tag-runs diva-gather-runs diva-all-runs
.PHONY: diva-theorem-gather diva-theorem-hallway diva-theorem-disperse diva-theorem-pursuit diva-theorem-runs
.PHONY: diva-top4-gh-packed diva-top4-dp-packed diva-top4-priority-packed diva-top6-priority-packed
.PHONY: qmix-gather qmix-hallway qmix-disperse qmix-runs
.PHONY: qmix-top5-gather qmix-top5-hallway qmix-top5-disperse qmix-top5-runs
.PHONY: vdn-top5-gather vdn-top5-hallway vdn-top5-disperse vdn-top5-runs
.PHONY: qtran-top5-gather qtran-top5-hallway qtran-top5-disperse qtran-top5-runs
.PHONY: baseline-top5-gather baseline-top5-hallway baseline-top5-disperse baseline-top5-runs
.PHONY: baseline-top4-disperse-qmix-vdn baseline-top4-hallway-qmix-vdn baseline-top4-qtran-hd baseline-top4-hd-packed
.PHONY: pursuit-then-baseline-top5-runs

SHELL := /bin/bash
PYTHON ?= python
USE_CUDA ?= True
DIVA_THEOREM_CONFIG ?= diva_bounded_sigmoid_qmix_DIVA
DIVA_TOP4_CONFIG ?= diva_bounded_sigmoid_qmix_DIVA_scale1_capacity64_gate2
DIVA_TOP6_CONFIG ?= diva_bounded_sigmoid_qmix_DIVA_scale1_capacity64_gate2
QMIX_CONFIG ?= qmix
VDN_CONFIG ?= vdn
QTRAN_CONFIG ?= qtran
GPU_COUNT ?= 4
MAX_JOBS_PER_GPU ?= 2

PURSUIT_SEEDS ?= 1 2 3 4 5 6 7 8
GATHER_TOP5_SEEDS ?= 2 4 3 1 8
HALLWAY_TOP5_SEEDS ?= 1 5 3 8 2
DISPERSE_TOP5_SEEDS ?= 8 5 3 1 7
GATHER_TOP4_SEEDS ?= 2 4 3 1
HALLWAY_TOP4_SEEDS ?= 1 5 3 8
DISPERSE_TOP4_SEEDS ?= 8 5 3 1

# Top-4 seeds for the latest bounded-sigmoid DIVA mixer,
# ranked by final held-out performance from `results_DIVA softplus/sacred`.
# Ties are broken by smaller seed id for determinism.
DIVA_GATHER_TOP4_SEEDS ?= 8 2 4 5
DIVA_HALLWAY_TOP4_SEEDS ?= 1 5 3 2
DIVA_DISPERSE_TOP4_SEEDS ?= 2 3 6 5
DIVA_PURSUIT_TOP4_SEEDS ?= 4 1 2 3

DIVA_GATHER_TOP6_SEEDS ?= 8 2 4 5 3 7
DIVA_HALLWAY_TOP6_SEEDS ?= 1 5 3 2 8 7
DIVA_DISPERSE_TOP6_SEEDS ?= 2 3 6 5 1 8

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

define run_two_baselines_for_env
set -eu; \
env_cfg="$(1)"; \
seeds="$(2)"; \
config_a="$(3)"; \
config_b="$(4)"; \
seed_arr=($$seeds); \
if [ $${#seed_arr[@]} -ne 4 ]; then \
	echo "Expected exactly 4 seeds for $$env_cfg, got: $${#seed_arr[@]}"; \
	exit 1; \
fi; \
echo "[batch] env=$$env_cfg configs=$$config_a,$$config_b"; \
for gpu in 0 1 2 3; do \
	seed=$${seed_arr[$$gpu]}; \
	echo "  gpu$$gpu -> seed=$$seed ($$config_a, $$config_b)"; \
	CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config_a --env-config=$$env_cfg with seed=$$seed use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config_b --env-config=$$env_cfg with seed=$$seed use_cuda=$(USE_CUDA) & \
	:; \
done; \
wait
endef

define run_one_baseline_for_two_envs
	set -eu; \
	config="$(1)"; \
env_a="$(2)"; \
seeds_a="$(3)"; \
env_b="$(4)"; \
seeds_b="$(5)"; \
seed_arr_a=($$seeds_a); \
seed_arr_b=($$seeds_b); \
if [ $${#seed_arr_a[@]} -ne 4 ]; then \
	echo "Expected exactly 4 seeds for $$env_a, got: $${#seed_arr_a[@]}"; \
	exit 1; \
fi; \
if [ $${#seed_arr_b[@]} -ne 4 ]; then \
	echo "Expected exactly 4 seeds for $$env_b, got: $${#seed_arr_b[@]}"; \
	exit 1; \
fi; \
echo "[batch] config=$$config envs=$$env_a,$$env_b"; \
for gpu in 0 1 2 3; do \
	seed_a=$${seed_arr_a[$$gpu]}; \
	seed_b=$${seed_arr_b[$$gpu]}; \
	echo "  gpu$$gpu -> $$env_a seed=$$seed_a, $$env_b seed=$$seed_b ($$config)"; \
	CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_a with seed=$$seed_a use_cuda=$(USE_CUDA) & \
	CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_b with seed=$$seed_b use_cuda=$(USE_CUDA) & \
	:; \
	done; \
	wait
endef

define run_one_config_for_four_envs_priority
	set -eu; \
	config="$(1)"; \
	env_a="$(2)"; \
	seeds_a="$(3)"; \
	env_b="$(4)"; \
	seeds_b="$(5)"; \
	env_c="$(6)"; \
	seeds_c="$(7)"; \
	env_d="$(8)"; \
	seeds_d="$(9)"; \
	seed_arr_a=($$seeds_a); \
	seed_arr_b=($$seeds_b); \
	seed_arr_c=($$seeds_c); \
	seed_arr_d=($$seeds_d); \
	for spec in "$$env_a:$${#seed_arr_a[@]}" "$$env_b:$${#seed_arr_b[@]}" "$$env_c:$${#seed_arr_c[@]}" "$$env_d:$${#seed_arr_d[@]}"; do \
		env_name="$${spec%%:*}"; \
		env_count="$${spec##*:}"; \
		if [ "$$env_count" -ne 4 ]; then \
			echo "Expected exactly 4 seeds for $$env_name, got: $$env_count"; \
			exit 1; \
		fi; \
	done; \
	echo "[priority-batch] config=$$config order=$$env_a,$$env_b then $$env_c,$$env_d"; \
	for gpu in 0 1 2 3; do \
		seed_a=$${seed_arr_a[$$gpu]}; \
		seed_b=$${seed_arr_b[$$gpu]}; \
		seed_c=$${seed_arr_c[$$gpu]}; \
		seed_d=$${seed_arr_d[$$gpu]}; \
		echo "  gpu$$gpu -> $$env_a seed=$$seed_a, $$env_b seed=$$seed_b, then $$env_c seed=$$seed_c, $$env_d seed=$$seed_d"; \
		( set -eu; \
			CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_a with seed=$$seed_a use_cuda=$(USE_CUDA) & \
			CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_b with seed=$$seed_b use_cuda=$(USE_CUDA) & \
			wait -n; \
			CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_c with seed=$$seed_c use_cuda=$(USE_CUDA) & \
			wait -n; \
			CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_d with seed=$$seed_d use_cuda=$(USE_CUDA) & \
			wait; \
		) & \
		:; \
	done; \
	wait
endef

define run_one_config_for_three_envs_top6_priority
	set -eu; \
	config="$(1)"; \
	env_a="$(2)"; \
	seeds_a="$(3)"; \
	env_b="$(4)"; \
	seeds_b="$(5)"; \
	env_c="$(6)"; \
	seeds_c="$(7)"; \
	seed_arr_a=($$seeds_a); \
	seed_arr_b=($$seeds_b); \
	seed_arr_c=($$seeds_c); \
	for spec in "$$env_a:$${#seed_arr_a[@]}" "$$env_b:$${#seed_arr_b[@]}" "$$env_c:$${#seed_arr_c[@]}"; do \
		env_name="$${spec%%:*}"; \
		env_count="$${spec##*:}"; \
		if [ "$$env_count" -ne 6 ]; then \
			echo "Expected exactly 6 seeds for $$env_name, got: $$env_count"; \
			exit 1; \
		fi; \
	done; \
	echo "[top6-priority] config=$$config order=$$env_a,$$env_b,$$env_c"; \
	for gpu in 0 1 2 3; do \
		case "$$gpu" in \
			0) jobs=("$$env_a:$${seed_arr_a[0]}" "$$env_b:$${seed_arr_b[0]}" "$$env_a:$${seed_arr_a[4]}" "$$env_b:$${seed_arr_b[4]}" "$$env_c:$${seed_arr_c[0]}");; \
			1) jobs=("$$env_a:$${seed_arr_a[1]}" "$$env_b:$${seed_arr_b[1]}" "$$env_a:$${seed_arr_a[5]}" "$$env_b:$${seed_arr_b[5]}" "$$env_c:$${seed_arr_c[1]}");; \
			2) jobs=("$$env_a:$${seed_arr_a[2]}" "$$env_b:$${seed_arr_b[2]}" "$$env_c:$${seed_arr_c[2]}" "$$env_c:$${seed_arr_c[4]}");; \
			3) jobs=("$$env_a:$${seed_arr_a[3]}" "$$env_b:$${seed_arr_b[3]}" "$$env_c:$${seed_arr_c[3]}" "$$env_c:$${seed_arr_c[5]}");; \
		esac; \
		echo "  gpu$$gpu queue: $${jobs[*]}"; \
		( set -eu; \
			queue=("$${jobs[@]}"); \
			next_idx=0; \
			pids=(); \
			while [ $$next_idx -lt $${#queue[@]} ] && [ $${#pids[@]} -lt 2 ]; do \
				job="$${queue[$$next_idx]}"; \
				env_name="$${job%%:*}"; \
				seed="$${job##*:}"; \
				echo "    launch gpu$$gpu -> $$env_name seed=$$seed"; \
				CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_name with seed=$$seed use_cuda=$(USE_CUDA) & \
				pids+=($$!); \
				next_idx=$$((next_idx + 1)); \
			done; \
			while [ $${#pids[@]} -gt 0 ]; do \
				wait -n; \
				alive=(); \
				for pid in "$${pids[@]}"; do \
					if kill -0 "$$pid" 2>/dev/null; then \
						alive+=("$$pid"); \
					fi; \
				done; \
				pids=("$${alive[@]}"); \
				while [ $$next_idx -lt $${#queue[@]} ] && [ $${#pids[@]} -lt 2 ]; do \
					job="$${queue[$$next_idx]}"; \
					env_name="$${job%%:*}"; \
					seed="$${job##*:}"; \
					echo "    backfill gpu$$gpu -> $$env_name seed=$$seed"; \
					CUDA_VISIBLE_DEVICES=$$gpu $(PYTHON) src/main.py --config=$$config --env-config=$$env_name with seed=$$seed use_cuda=$(USE_CUDA) & \
					pids+=($$!); \
					next_idx=$$((next_idx + 1)); \
				done; \
			done; \
		) & \
		:; \
	done; \
	wait
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

# Top-4 seeds packed at 2 jobs per GPU:
# - gather:   2 4 3 1
# - hallway:  1 5 3 8
# - disperse: 8 5 3 1
#
# This layout launches 8 jobs in each batch:
# - GPU 0 gets the best seed for the env
# - GPU 1 gets the second seed
# - GPU 2 gets the third seed
# - GPU 3 gets the fourth seed
# and the batches run in this order:
# 1. disperse with QMIX + VDN
# 2. hallway with QMIX + VDN
# 3. QTRAN on both disperse and hallway together

baseline-top4-disperse-qmix-vdn:
	@$(call run_two_baselines_for_env,disperse,$(DISPERSE_TOP4_SEEDS),$(QMIX_CONFIG),$(VDN_CONFIG))

baseline-top4-hallway-qmix-vdn:
	@$(call run_two_baselines_for_env,hallway,$(HALLWAY_TOP4_SEEDS),$(QMIX_CONFIG),$(VDN_CONFIG))

baseline-top4-qtran-hd:
	@$(call run_one_baseline_for_two_envs,$(QTRAN_CONFIG),disperse,$(DISPERSE_TOP4_SEEDS),hallway,$(HALLWAY_TOP4_SEEDS))

baseline-top4-hd-packed:
	$(MAKE) baseline-top4-disperse-qmix-vdn PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG) VDN_CONFIG=$(VDN_CONFIG)
	$(MAKE) baseline-top4-hallway-qmix-vdn PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG) VDN_CONFIG=$(VDN_CONFIG)
	$(MAKE) baseline-top4-qtran-hd PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QTRAN_CONFIG=$(QTRAN_CONFIG)

# Top-4 DIVA packed at 2 jobs per GPU using the current best-guess mixer config:
# - gather:   8 2 4 5
# - hallway:  1 5 3 2
# - disperse: 2 3 6 5
# - pursuit:  4 1 2 3
#
# Launch order matches the requested environment priority:
# - each GPU starts with gather + hallway
# - when one of those finishes, it backfills disperse
# - when another slot frees up, it backfills pursuit
# Within each environment, GPU 0 gets the best seed, GPU 1 the second, etc.

diva-top4-gh-packed:
	@$(call run_one_baseline_for_two_envs,$(DIVA_TOP4_CONFIG),gather,$(DIVA_GATHER_TOP4_SEEDS),hallway,$(DIVA_HALLWAY_TOP4_SEEDS))

diva-top4-dp-packed:
	@$(call run_one_baseline_for_two_envs,$(DIVA_TOP4_CONFIG),disperse,$(DIVA_DISPERSE_TOP4_SEEDS),pursuit,$(DIVA_PURSUIT_TOP4_SEEDS))

diva-top4-priority-packed:
	@$(call run_one_config_for_four_envs_priority,$(DIVA_TOP4_CONFIG),gather,$(DIVA_GATHER_TOP4_SEEDS),hallway,$(DIVA_HALLWAY_TOP4_SEEDS),disperse,$(DIVA_DISPERSE_TOP4_SEEDS),pursuit,$(DIVA_PURSUIT_TOP4_SEEDS))

# Top-6 DIVA packed at 2 jobs per GPU using the current best-guess mixer config:
# - gather:   8 2 4 5 3 7
# - hallway:  1 5 3 2 8 7
# - disperse: 2 3 6 5 1 8
#
# Priority order is gather, then hallway, then disperse.
# The scheduler keeps up to 2 jobs per GPU by backfilling from each GPU's queue
# until that queue is exhausted. The very end can have fewer than 8 total jobs
# because there are only 18 runs overall.

diva-top6-priority-packed:
	@$(call run_one_config_for_three_envs_top6_priority,$(DIVA_TOP6_CONFIG),gather,$(DIVA_GATHER_TOP6_SEEDS),hallway,$(DIVA_HALLWAY_TOP6_SEEDS),disperse,$(DIVA_DISPERSE_TOP6_SEEDS))

pursuit-then-baseline-top5-runs:
	$(MAKE) diva-theorem-pursuit PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) DIVA_THEOREM_CONFIG=$(DIVA_THEOREM_CONFIG)
	$(MAKE) baseline-top5-runs PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA) QMIX_CONFIG=$(QMIX_CONFIG) VDN_CONFIG=$(VDN_CONFIG) QTRAN_CONFIG=$(QTRAN_CONFIG)

run_baselines: 
	for f in runs_baselines/*.job; do sbatch $$f; done
