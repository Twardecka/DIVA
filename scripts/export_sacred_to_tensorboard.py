#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Sacred run info.json scalars into per-run TensorBoard logs."
    )
    parser.add_argument(
        "--sacred-dir",
        type=Path,
        default=Path("results/sacred"),
        help="Directory containing Sacred run folders.",
    )
    parser.add_argument(
        "--tb-dir",
        type=Path,
        default=Path("results/tb_logs_sacred"),
        help="Output directory for TensorBoard event files.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete existing exported event files before writing new ones.",
    )
    parser.add_argument(
        "--group-by",
        choices=("none", "env"),
        default="env",
        help="Optional directory grouping for exported runs.",
    )
    return parser.parse_args()


def scalar_to_float(value):
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def metric_series(info):
    for key, values in info.items():
        if key.endswith("_T"):
            continue
        steps = info.get(f"{key}_T")
        if not isinstance(values, list) or not isinstance(steps, list):
            continue
        if len(values) != len(steps):
            continue
        pairs = []
        for step, value in zip(steps, values):
            try:
                step_int = int(step)
            except (TypeError, ValueError):
                continue
            scalar = scalar_to_float(value)
            if scalar is None:
                continue
            pairs.append((step_int, scalar))
        if pairs:
            yield key, pairs


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def sanitize_path_component(value):
    safe = str(value).strip().replace("/", "_")
    return safe or "unknown"


def infer_status(run_data, max_logged_step):
    status = run_data.get("status", "UNKNOWN")
    result = run_data.get("result")
    if status == "COMPLETED":
        return "completed"
    if max_logged_step is None:
        return "no-metrics"
    if result is not None:
        return "completed"
    return "likely-running-or-exited-uncleanly"


def maybe_purge(path):
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            maybe_purge(child)
            child.rmdir()
        else:
            child.unlink()


def main():
    args = parse_args()
    sacred_dir = args.sacred_dir.resolve()
    tb_dir = args.tb_dir.resolve()

    if args.purge and tb_dir.exists():
        maybe_purge(tb_dir)

    tb_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted((p for p in sacred_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    summaries = []

    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        info_path = run_dir / "info.json"
        run_path = run_dir / "run.json"
        cout_path = run_dir / "cout.txt"

        if not (config_path.exists() and info_path.exists() and run_path.exists()):
            continue

        config = load_json(config_path)
        info = load_json(info_path)
        run_data = load_json(run_path)

        env_name = config.get("env") or config.get("env_args", {}).get("key") or "unknown_env"
        alg_name = config.get("name", "unknown_alg")
        seed = config.get("seed", "unknown_seed")
        run_id = int(run_dir.name)

        run_tb_dir = tb_dir
        if args.group_by == "env":
            run_tb_dir = run_tb_dir / sanitize_path_component(env_name)
        run_tb_dir = run_tb_dir / f"run_{run_id:02d}__{alg_name}__{env_name}__seed{seed}"
        run_tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_tb_dir))

        metric_count = 0
        max_logged_step = None
        for key, pairs in metric_series(info):
            metric_count += 1
            for step, value in pairs:
                writer.add_scalar(key, value, step)
                if max_logged_step is None or step > max_logged_step:
                    max_logged_step = step

        writer.add_text("run/config", json.dumps(config, indent=2), 0)
        writer.add_text("run/metadata", json.dumps(run_data, indent=2), 0)
        writer.flush()
        writer.close()

        saved_model = False
        last_saved_path = None
        if cout_path.exists():
            for line in cout_path.read_text().splitlines():
                if "Saving models to " in line:
                    saved_model = True
                    last_saved_path = line.split("Saving models to ", 1)[1].strip()

        summaries.append(
            {
                "run_id": run_id,
                "alg": alg_name,
                "env": env_name,
                "seed": seed,
                "metrics": metric_count,
                "max_logged_step": max_logged_step,
                "status": infer_status(run_data, max_logged_step),
                "saved_model": saved_model,
                "last_saved_path": last_saved_path,
                "tb_dir": str(run_tb_dir),
            }
        )

    summaries.sort(key=lambda item: item["run_id"])

    print(f"Exported {len(summaries)} runs to {tb_dir}")
    print(f"Grouping: {args.group_by}")
    print()
    for item in summaries:
        print(
            "run={run_id:>2} alg={alg:<18} env={env:<8} seed={seed:<2} "
            "step={step:<7} metrics={metrics:<2} model={model:<3} status={status}".format(
                run_id=item["run_id"],
                alg=item["alg"],
                env=item["env"],
                seed=item["seed"],
                step=item["max_logged_step"] if item["max_logged_step"] is not None else "-",
                metrics=item["metrics"],
                model="yes" if item["saved_model"] else "no",
                status=item["status"],
            )
        )


if __name__ == "__main__":
    main()
