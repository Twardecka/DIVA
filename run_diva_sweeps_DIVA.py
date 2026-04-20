#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RUNS = [
    ("diva_softmax_DIVA", "tag"),
    ("diva_softplus_DIVA", "tag"),
    ("diva_softmax_DIVA", "gather"),
    ("diva_softplus_DIVA", "gather"),
]


def build_command(python_bin, alg, env, seed, save_model, use_tensorboard):
    return [
        python_bin,
        "src/main.py",
        f"--config={alg}",
        f"--env-config={env}",
        "with",
        f"seed={seed}",
        f"save_model={str(save_model)}",
        f"use_tensorboard={str(use_tensorboard)}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Run DIVA sweeps for tag/pursuit and gather.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to use for training.")
    parser.add_argument("--seed-start", type=int, default=1, help="First seed in the sweep.")
    parser.add_argument("--seed-end", type=int, default=10, help="Last seed in the sweep.")
    parser.add_argument("--save-model", action="store_true", help="Enable model checkpoint saving.")
    parser.add_argument("--use-tensorboard", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    log_dir = root / "results" / "diva_sweeps" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for alg, env in DEFAULT_RUNS:
        for seed in range(args.seed_start, args.seed_end + 1):
            command = build_command(
                args.python_bin,
                alg,
                env,
                seed,
                args.save_model,
                args.use_tensorboard,
            )
            log_path = log_dir / f"{alg}__{env}__seed{seed}.log"
            print(f"Prepared: {alg} on {env} with seed={seed}")
            print(" ".join(command))
            print(f"log -> {log_path}")

            if args.dry_run:
                continue

            with open(log_path, "w") as log_file:
                subprocess.run(command, cwd=root, stdout=log_file, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
