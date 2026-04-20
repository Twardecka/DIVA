#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source /home/ludwika/miniconda3/bin/activate gacg

mkdir -p results/diva_sweeps/logs

configs=(
  "diva_softmax_DIVA tag"
  "diva_softplus_DIVA tag"
  "diva_softmax_DIVA gather"
  "diva_softplus_DIVA gather"
)

for pair in "${configs[@]}"; do
  set -- $pair
  alg="$1"
  env="$2"

  for seed in $(seq 1 10); do
    log_path="results/diva_sweeps/logs/${alg}__${env}__seed${seed}.log"
    echo "Running ${alg} on ${env} with seed=${seed}"
    python src/main.py \
      --config="${alg}" \
      --env-config="${env}" \
      with seed="${seed}" save_model=False use_tensorboard=False \
      > "${log_path}" 2>&1
  done
done
