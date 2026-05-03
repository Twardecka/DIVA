#!/usr/bin/env python3

"""Generate one SLURM job file per discovery run."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ACCOUNT = "prasanna_1363"
DEFAULT_PARTITION = "main"
DEFAULT_RUNS_DIRECTORY = "runs_discovery"
DEFAULT_CONDA_ENV = "gacg"
DEFAULT_MODULES = ["legacy/CentOS7", "gcc/11.3.0", "git/2.36.1"]
DEFAULT_PYTHON_BIN = "python3"
DEFAULT_CONFIGS = ["gacg"]
DEFAULT_ENVS = ["pogema"]
DEFAULT_SEEDS = [0]
DEFAULT_USE_CUDA = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SLURM discovery jobs.")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--runs-directory", default=DEFAULT_RUNS_DIRECTORY)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--cpus-per-task", type=int, default=64)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--time", default="48:00:00")
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        default=None,
        help="Module to load. Repeat to add more modules.",
    )
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_CUDA,
        help="Set use_cuda in the generated training command.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Sacred override to append after `with`. Repeat as needed.",
    )
    return parser.parse_args()


def build_header(args: argparse.Namespace) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --account={args.account}",
        f"#SBATCH --partition={args.partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time}",
        "",
        f"conda activate {args.conda_env}",
    ]

    modules = args.modules if args.modules is not None else DEFAULT_MODULES
    if modules:
        lines.append(f"module load {' '.join(modules)}")

    lines.extend(
        [
            "",
            'echo "Starting parallel job script"',
        ]
    )
    return lines


def build_command(
    config: str,
    env: str,
    seed: int,
    args: argparse.Namespace,
) -> str:
    use_cuda = "True" if args.use_cuda else "False"
    parts = [
        args.python_bin,
        "src/main.py",
        f"--config={config}",
        f"--env-config={env}",
        "with",
        f"seed={seed}",
        f"use_cuda={use_cuda}",
    ]
    parts.extend(args.extra_override)
    return " ".join(parts)


def build_job_script(
    config: str,
    env: str,
    seed: int,
    args: argparse.Namespace,
) -> str:
    lines = build_header(args)
    lines.append(build_command(config, env, seed, args))
    lines.append("")
    return "\n".join(lines)


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="ascii")
    path.chmod(0o755)


def clean_previous_jobs(runs_dir: Path) -> None:
    for path in runs_dir.glob("run_*.job"):
        path.unlink()

    submit_path = runs_dir / "submit_all.sh"
    if submit_path.exists():
        submit_path.unlink()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    runs_dir = root / args.runs_directory
    runs_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_jobs(runs_dir)

    submit_lines = ["#!/bin/bash", ""]

    job_index = 1
    for seed in args.seeds:
        for env in args.envs:
            for config in args.configs:
                job_path = runs_dir / f"run_{job_index}.job"
                job_script = build_job_script(config, env, seed, args)
                write_file(job_path, job_script)
                submit_lines.append(f"sbatch {job_path.relative_to(root)}")
                job_index += 1

    submit_path = runs_dir / "submit_all.sh"
    write_file(submit_path, "\n".join(submit_lines) + "\n")

    total_runs = len(args.seeds) * len(args.envs) * len(args.configs)
    print(f"Wrote {total_runs} job files to {runs_dir}")
    print(f"Submit all with: {submit_path}")


if __name__ == "__main__":
    main()
