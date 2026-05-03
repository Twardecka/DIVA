#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path


ACCOUNT = "prasanna_1363"
PARTITION = "main"
ACTIVATE_LINE = "source /home/ludwika/miniconda3/bin/activate gacg"
MODULE_LINE = "module load legacy/CentOS7 gcc/11.3.0 git/2.36.1"
PYTHON_BIN = "/home/ludwika/miniconda3/envs/gacg/bin/python"
RUNS_DIR = Path("runs_baselines")
SEEDS = [1, 3, 5, 8]
ENVS = ["hallway", "disperse", "sensor"]
BASELINE_ENVS = ["sensor"]
BASELINES = ["qmix", "qtran", "vdn"]
T_MAX_OVERRIDE = "t_max=1005000"


def build_job_specs():
    jobs = []

    for env in ENVS:
        for seed in SEEDS:
            jobs.append(
                {
                    "group": "diva_vdn_like_1m",
                    "algo": "diva_bounded_sigmoid_vdn_DIVA",
                    "env": env,
                    "seed": seed,
                    "overrides": [T_MAX_OVERRIDE],
                }
            )

    for env in BASELINE_ENVS:
        for algo in BASELINES:
            for seed in SEEDS:
                jobs.append(
                    {
                        "group": "baselines_1m",
                        "algo": algo,
                        "env": env,
                        "seed": seed,
                        "overrides": [T_MAX_OVERRIDE],
                    }
                )

    for seed in SEEDS:
        jobs.append(
            {
                "group": "sensor_diva_gpu_version_1m",
                "algo": "diva_bounded_sigmoid_qmix_DIVA_vscale1_clean",
                "env": "sensor",
                "seed": seed,
                "overrides": [T_MAX_OVERRIDE],
            }
        )

    return jobs


def build_command(job):
    parts = [
        PYTHON_BIN,
        "src/main.py",
        f"--config={job['algo']}",
        f"--env-config={job['env']}",
        "with",
        f"seed={job['seed']}",
        "use_cuda=False",
    ]
    parts.extend(job["overrides"])
    return " ".join(parts)


def build_job_script(job):
    lines = [
        "#!/bin/bash",
        f"#SBATCH --account={ACCOUNT}",
        f"#SBATCH --partition={PARTITION}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=64",
        "#SBATCH --mem=128G",
        "#SBATCH --time=48:00:00",
        "",
        ACTIVATE_LINE,
        MODULE_LINE,
        "",
        'echo "Starting parallel job script"',
        build_command(job),
        "",
    ]
    return "\n".join(lines)


def clean_runs_dir(runs_dir: Path):
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)
        return

    for path in runs_dir.iterdir():
        if path.is_file():
            path.unlink()


def write_text(path: Path, text: str):
    path.write_text(text, encoding="ascii")
    path.chmod(0o755)


def write_manifest(runs_dir: Path, jobs):
    manifest_path = runs_dir / "job_manifest.csv"
    with manifest_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["job_file", "group", "algo", "env", "seed", "command"])
        for index, job in enumerate(jobs, start=1):
            writer.writerow(
                [
                    f"run_{index}.job",
                    job["group"],
                    job["algo"],
                    job["env"],
                    job["seed"],
                    build_command(job),
                ]
            )


def write_readme(runs_dir: Path, jobs):
    counts = {}
    for job in jobs:
        counts[job["group"]] = counts.get(job["group"], 0) + 1

    lines = [
        "runs_baselines",
        "==============",
        "",
        "This bundle contains CPU-only 1M jobs with seeds 1, 3, 5, 8.",
        "",
        "Groups:",
        f"- diva_vdn_like_1m: {counts.get('diva_vdn_like_1m', 0)} jobs",
        "  config: diva_bounded_sigmoid_vdn_DIVA",
        "  envs: hallway, disperse, sensor",
        "",
        f"- baselines_1m: {counts.get('baselines_1m', 0)} jobs",
        "  configs: qmix, qtran, vdn",
        "  envs: sensor only",
        "",
        f"- sensor_diva_gpu_version_1m: {counts.get('sensor_diva_gpu_version_1m', 0)} jobs",
        "  config: diva_bounded_sigmoid_qmix_DIVA_vscale1_clean",
        "  env: sensor",
        "",
        "All jobs use the explicit Sacred override:",
        f"- {T_MAX_OVERRIDE}",
        "",
        "See job_manifest.csv for the exact mapping from run_N.job to config/env/seed.",
    ]
    (runs_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def write_submit_script(runs_dir: Path, total_jobs: int):
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for index in range(1, total_jobs + 1):
        lines.append(f"sbatch runs_baselines/run_{index}.job")
    write_text(runs_dir / "submit_all.sh", "\n".join(lines) + "\n")


def main():
    root = Path(__file__).resolve().parents[1]
    runs_dir = root / RUNS_DIR
    jobs = build_job_specs()

    clean_runs_dir(runs_dir)

    for index, job in enumerate(jobs, start=1):
        job_path = runs_dir / f"run_{index}.job"
        write_text(job_path, build_job_script(job))

    write_submit_script(runs_dir, len(jobs))
    write_manifest(runs_dir, jobs)
    write_readme(runs_dir, jobs)

    print(f"Wrote {len(jobs)} jobs to {runs_dir}")
    print(f"Manifest: {runs_dir / 'job_manifest.csv'}")


if __name__ == "__main__":
    main()
