.PHONY: all diva-tag-softmax diva-tag-softplus diva-gather-softmax diva-gather-softplus diva-tag-runs diva-gather-runs diva-all-runs

PYTHON ?= python
USE_CUDA ?= True

all:
	clear
	# Priority order:
	# 1. tag / softmax_DIVA
	# 2. tag / softplus_DIVA
	# 3. gather / softmax_DIVA
	# 4. gather / softplus_DIVA
	#
	# Each target launches 8 jobs total:
	# 2 jobs on each of 4 GPUs, using seeds 1..8.
	#
	# Example:
	# make diva-tag-softmax PYTHON=/home/ludwika/miniconda3/envs/gacg/bin/python
	# make diva-tag-runs PYTHON=/home/ludwika/miniconda3/envs/gacg/bin/python
	# make diva-all-runs PYTHON=/home/ludwika/miniconda3/envs/gacg/bin/python

diva-tag-softmax:
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=1 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=2 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=3 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=4 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=5 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=6 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=7 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=tag with seed=8 use_cuda=$(USE_CUDA) &

diva-tag-softplus:
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=1 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=2 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=3 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=4 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=5 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=6 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=7 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=tag with seed=8 use_cuda=$(USE_CUDA) &

diva-gather-softmax:
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=1 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=2 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=3 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=4 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=5 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=6 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=7 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softmax_DIVA --env-config=gather with seed=8 use_cuda=$(USE_CUDA) &

diva-gather-softplus:
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=1 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=2 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=3 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=4 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=5 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=6 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=7 use_cuda=$(USE_CUDA) &
	CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/main.py --config=diva_softplus_DIVA --env-config=gather with seed=8 use_cuda=$(USE_CUDA) &

diva-tag-runs:
	$(MAKE) diva-tag-softmax PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)
	$(MAKE) diva-tag-softplus PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)

diva-gather-runs:
	$(MAKE) diva-gather-softmax PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)
	$(MAKE) diva-gather-softplus PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)

diva-all-runs:
	$(MAKE) diva-tag-runs PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)
	$(MAKE) diva-gather-runs PYTHON=$(PYTHON) USE_CUDA=$(USE_CUDA)
