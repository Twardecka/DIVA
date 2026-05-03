#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TB_ROOT = Path("results_sensor/tb_logs")
OUTPUT_DIR = Path("artifacts/sensor_plots")
TARGET_SEEDS = {1, 3, 5, 8}
SELECTION_METRIC = "test_scaned_mean"
RUN_DIR_PATTERN = re.compile(
    r"^(?P<algorithm>.+)-seed(?P<seed>\d+)__(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
)

DISPLAY_NAMES = {
    "diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5": "DIVA v3",
    "qmix": "QMIX",
    "qtran": "QTRAN",
    "vdn": "VDN",
}

COLOR_MAP = {
    "diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5": "#d62728",
    "qmix": "#1f77b4",
    "qtran": "#9467bd",
    "vdn": "#2ca02c",
}

LINESTYLE_MAP = {
    "diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5": "-",
    "qmix": "-.",
    "qtran": (0, (3, 1, 1, 1)),
    "vdn": ":",
}

PLOT_ORDER = [
    "diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5",
    "qmix",
    "vdn",
    "qtran",
]

METRICS = {
    "test_return_mean": "Test return mean",
    "test_scaned_mean": "Test scanned mean",
}

YLIMS = {
    "test_return_mean": (-5.0, 8.5),
    "test_scaned_mean": (0.0, 8.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot clean sensor runs only, filtering mixed tb_logs by sensor-specific metrics."
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.75,
        help="EMA smoothing weight in [0, 1). Higher is smoother.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=505_000,
        help="Discard points above this step count.",
    )
    return parser.parse_args()


def ema_smooth(data, weight):
    if len(data) == 0:
        return data
    smoothed = []
    last = data[0]
    for point in data:
        last = last * weight + (1.0 - weight) * point
        smoothed.append(last)
    return np.array(smoothed)


def load_accumulator(event_file):
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    return accumulator


def has_scalar(accumulator, metric_name):
    tags = accumulator.Tags()
    return "scalars" in tags and metric_name in tags["scalars"]


def extract_scalar(accumulator, metric_name, max_steps):
    if not has_scalar(accumulator, metric_name):
        return None
    events = accumulator.Scalars(metric_name)
    steps = np.array([event.step for event in events])
    values = np.array([event.value for event in events])
    mask = steps <= max_steps
    return steps[mask], values[mask]


def select_clean_sensor_runs(max_steps):
    selected = {}
    for run_dir in sorted(path for path in TB_ROOT.iterdir() if path.is_dir()):
        match = RUN_DIR_PATTERN.match(run_dir.name)
        if not match:
            continue

        algorithm = match.group("algorithm")
        if algorithm not in DISPLAY_NAMES:
            continue

        seed = int(match.group("seed"))
        if seed not in TARGET_SEEDS:
            continue

        event_files = sorted(run_dir.glob("events.out.tfevents*"))
        if not event_files:
            continue

        event_file = event_files[-1]
        accumulator = load_accumulator(event_file)

        # Only true sensor runs contain the sensor-specific scanned metric.
        if not has_scalar(accumulator, SELECTION_METRIC):
            continue

        key = (algorithm, seed)
        timestamp = match.group("timestamp")
        current = selected.get(key)
        if current is None or timestamp > current["timestamp"]:
            selected[key] = {
                "algorithm": algorithm,
                "seed": seed,
                "timestamp": timestamp,
                "run_dir": run_dir,
                "accumulator": accumulator,
            }

    return selected


def collect_metric_runs(selected_runs, metric_name, max_steps):
    grouped = defaultdict(list)
    for run in selected_runs.values():
        data = extract_scalar(run["accumulator"], metric_name, max_steps)
        if data is None:
            continue
        grouped[run["algorithm"]].append(
            {
                "seed": run["seed"],
                "run_dir": run["run_dir"],
                "steps": data[0],
                "values": data[1],
            }
        )
    return grouped


def align_and_aggregate(runs, smoothing):
    min_len = min(len(run["values"]) for run in runs)
    steps = runs[0]["steps"][:min_len]
    values_array = np.stack([run["values"][:min_len] for run in runs], axis=0)
    median = ema_smooth(np.median(values_array, axis=0), smoothing)
    low = ema_smooth(np.percentile(values_array, 25, axis=0), smoothing)
    high = ema_smooth(np.percentile(values_array, 75, axis=0), smoothing)
    return steps, median, low, high


def plot_metric(axis, grouped_runs, metric_name, smoothing):
    for algorithm in PLOT_ORDER:
        runs = grouped_runs.get(algorithm)
        if not runs:
            continue
        steps, median, low, high = align_and_aggregate(runs, smoothing)
        axis.plot(
            steps,
            median,
            label=f"{DISPLAY_NAMES[algorithm]} (n={len(runs)})",
            color=COLOR_MAP[algorithm],
            linestyle=LINESTYLE_MAP[algorithm],
            linewidth=2.4,
        )
        axis.fill_between(steps, low, high, color=COLOR_MAP[algorithm], alpha=0.12)

    axis.set_title(METRICS[metric_name], fontsize=16, fontweight="bold")
    axis.set_xlabel("Timesteps", fontsize=13, fontweight="bold")
    axis.set_ylabel(METRICS[metric_name], fontsize=13, fontweight="bold")
    axis.set_xlim(0, 505_000)
    axis.set_ylim(*YLIMS[metric_name])
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.tick_params(axis="both", labelsize=11)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_clean_sensor_runs(args.max_steps)
    print("Selected clean sensor runs:")
    for key in sorted(selected):
        run = selected[key]
        print(f"- {run['algorithm']} seed={run['seed']}: {run['run_dir'].name}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True)
    created = []

    for axis, metric_name in zip(axes, METRICS):
        grouped = collect_metric_runs(selected, metric_name, args.max_steps)
        plot_metric(axis, grouped, metric_name, args.smoothing)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11, frameon=False)
    fig.suptitle("Sensor: clean 4-seed comparison", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    combined_path = OUTPUT_DIR / "sensor_clean_combined.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    created.append(combined_path)
    plt.close(fig)

    for metric_name in METRICS:
        metric_runs = collect_metric_runs(selected, metric_name, args.max_steps)
        fig, axis = plt.subplots(1, 1, figsize=(8.5, 6.5))
        plot_metric(axis, metric_runs, metric_name, args.smoothing)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="lower right", fontsize=10, frameon=False)
        fig.tight_layout()
        metric_path = OUTPUT_DIR / f"sensor_clean_{metric_name}.png"
        fig.savefig(metric_path, dpi=300, bbox_inches="tight")
        created.append(metric_path)
        plt.close(fig)

    print("Created plots:")
    for path in created:
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()
