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


ENV_ROOTS = {
    "gather": Path("results_all_gather/top4_by_diva_v1/tb_logs"),
    "hallway": Path("results_all_hallway/top4_by_diva_v1/tb_logs"),
    "disperse": Path("results_all_disperse/top4_by_diva_v1/tb_logs"),
}

DISPLAY_NAMES = {
    "diva_v1": "DIVA v1",
    "diva_v2": "DIVA v2",
    "qmix": "QMIX",
    "vdn": "VDN",
    "qtran": "QTRAN",
}

COLOR_MAP = {
    "diva_v1": "#d62728",
    "diva_v2": "#ff9896",
    "qmix": "#1f77b4",
    "vdn": "#2ca02c",
    "qtran": "#9467bd",
}

LINESTYLE_MAP = {
    "diva_v1": "-",
    "diva_v2": "--",
    "qmix": "-.",
    "vdn": ":",
    "qtran": (0, (3, 1, 1, 1)),
}

SOURCE_PREFERENCE = {
    "gather": {
        "diva_v1": "diva_v1",
        "diva_v2": "diva_v2",
        "qmix": "baseline_new",
        "vdn": "baseline_new",
    },
    "hallway": {
        "diva_v1": "diva_v1",
        "diva_v2": "diva_v2",
        "qmix": "baseline_old",
        "vdn": "baseline_old",
        "qtran": "baseline_old",
    },
    "disperse": {
        "diva_v1": "diva_v1",
        "diva_v2": "diva_v2",
        "qmix": "baseline_old",
        "vdn": "baseline_old",
        "qtran": "baseline_old",
    },
}

RUN_DIR_PATTERN = re.compile(
    r"^(?P<algorithm>[A-Za-z0-9_]+)-seed(?P<seed>\d+)(?:-(?P<source>[A-Za-z0-9_]+)-run(?P<run_id>\d+))?$"
)

METRIC_LABELS = {
    "test_battle_won_mean": "Test win rate",
    "test_return_mean": "Test return mean",
    "test_win_group_mean": "Test win group mean",
    "test_match_mean": "Test match mean",
    "test_ep_length_mean": "Test episode length mean",
}

PLOT_ORDER = ["diva_v1", "diva_v2", "qmix", "vdn", "qtran"]

YLIM_OVERRIDES = {
    ("hallway", "test_return_mean"): (0.0, 5.0),
    ("disperse", "test_return_mean"): (-4.0, 0.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard metrics for the top4_by_diva_v1 comparison folders."
    )
    parser.add_argument(
        "--env",
        choices=("all", "gather", "hallway", "disperse"),
        default="all",
        help="Which environment to plot.",
    )
    parser.add_argument(
        "--metric",
        default="test_battle_won_mean",
        help="Scalar metric to plot from TensorBoard logs.",
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
        default=1_000_000,
        help="Discard points above this step count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/top4_by_diva_v1_plots"),
        help="Directory where plot images will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also create one combined multi-panel figure across the selected environments.",
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


def extract_events(event_file, metric_name, max_steps):
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    tags = accumulator.Tags()
    if "scalars" not in tags or metric_name not in tags["scalars"]:
        return None

    events = accumulator.Scalars(metric_name)
    steps = np.array([event.step for event in events])
    values = np.array([event.value for event in events])
    mask = steps <= max_steps
    return steps[mask], values[mask]


def discover_runs(root_dir, metric_name, max_steps):
    runs = []
    for run_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        match = RUN_DIR_PATTERN.match(run_dir.name)
        if not match:
            continue

        event_files = sorted(run_dir.glob("events.out.tfevents*"))
        if not event_files:
            continue

        data = extract_events(event_files[-1], metric_name, max_steps)
        if data is None:
            continue

        runs.append(
            {
                "path": run_dir,
                "name": run_dir.name,
                "algorithm": match.group("algorithm"),
                "seed": int(match.group("seed")),
                "source": match.group("source"),
                "run_id": int(match.group("run_id")) if match.group("run_id") else None,
                "steps": data[0],
                "values": data[1],
            }
        )
    return runs


def select_canonical_runs(runs, env_name):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["algorithm"], run["seed"])].append(run)

    selected = []
    preferences = SOURCE_PREFERENCE.get(env_name, {})

    for (algorithm, _seed), candidates in sorted(grouped.items()):
        if len(candidates) == 1:
            selected.append(candidates[0])
            continue

        preferred_source = preferences.get(algorithm)
        if preferred_source is not None:
            matching = [run for run in candidates if run["source"] == preferred_source]
            if matching:
                selected.append(sorted(matching, key=lambda run: run["name"])[-1])
                continue

        no_source = [run for run in candidates if run["source"] is None]
        if no_source:
            selected.append(sorted(no_source, key=lambda run: run["name"])[-1])
        else:
            selected.append(sorted(candidates, key=lambda run: run["name"])[-1])

    return selected


def group_by_algorithm(runs):
    grouped = defaultdict(list)
    for run in sorted(runs, key=lambda item: (item["algorithm"], item["seed"], item["name"])):
        grouped[run["algorithm"]].append((run["steps"], run["values"]))
    return grouped


def align_and_aggregate(runs, smoothing):
    min_len = min(len(values) for _, values in runs)
    steps = runs[0][0][:min_len]
    values_array = np.stack([values[:min_len] for _, values in runs], axis=0)
    median = ema_smooth(np.median(values_array, axis=0), smoothing)
    low = ema_smooth(np.percentile(values_array, 25, axis=0), smoothing)
    high = ema_smooth(np.percentile(values_array, 75, axis=0), smoothing)
    return steps, median, low, high


def metric_ylim(env_name, metric_name, values):
    override = YLIM_OVERRIDES.get((env_name, metric_name))
    if override is not None:
        return override
    if metric_name == "test_battle_won_mean":
        return 0.0, 1.05
    if metric_name == "test_ep_length_mean":
        lower = min(values) if values else 0.0
        upper = max(values) if values else 1.0
        return max(0.0, lower - 0.5), upper + 0.5
    lower = min(values) if values else 0.0
    upper = max(values) if values else 1.0
    if lower == upper:
        pad = 0.1 if lower == 0 else abs(lower) * 0.1
        return lower - pad, upper + pad
    pad = 0.05 * (upper - lower)
    return lower - pad, upper + pad


def zoomed_winrate_ylim(values):
    lower = min(values) if values else 0.0
    upper = max(values) if values else 0.0
    if upper <= 0.0 and lower >= 0.0:
        return 0.0, 0.01
    if lower == upper:
        pad = max(0.01, upper * 0.15)
        return max(0.0, lower - pad), min(1.05, upper + pad)
    pad = max(0.01, 0.12 * (upper - lower))
    return max(0.0, lower - pad), min(1.05, upper + pad)


def plot_algorithm_series(ax, key, steps, center, low, high, label_override=None):
    label = DISPLAY_NAMES.get(key, key)
    color = COLOR_MAP.get(key)
    linestyle = LINESTYLE_MAP.get(key, "-")

    ax.plot(
        steps,
        center,
        label=label_override or label,
        color=color,
        linewidth=2.5,
        linestyle=linestyle,
    )
    ax.fill_between(steps, low, high, color=color, alpha=0.16)

def compute_plot_series(grouped_runs, smoothing):
    series = {}
    all_plot_values = []
    for key in PLOT_ORDER + [key for key in sorted(grouped_runs) if key not in PLOT_ORDER]:
        if key not in grouped_runs:
            continue
        runs = grouped_runs[key]
        if len(runs) == 1:
            steps, values = runs[0]
            center = ema_smooth(values, smoothing)
            low = center.copy()
            high = center.copy()
        else:
            steps, center, low, high = align_and_aggregate(runs, smoothing)

        series[key] = (steps, center, low, high)
        all_plot_values.extend(low.tolist())
        all_plot_values.extend(high.tolist())
    return series, all_plot_values


def style_axis(ax, env_name, metric_name, values, show_ylabel):
    xticks = [0, 200_000, 400_000, 600_000, 800_000, 1_000_000]
    xticklabels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    ymin, ymax = metric_ylim(env_name, metric_name, values)
    ax.set_xlim(0, 1_000_000)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel("Timesteps (in millions)", fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name), fontweight="bold")
    ax.set_title(env_name.capitalize(), fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_env(env_name, grouped_runs, metric_name, smoothing, output_dir, show):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    series, all_plot_values = compute_plot_series(grouped_runs, smoothing)
    for key, (steps, center, low, high) in series.items():
        plot_algorithm_series(ax, key, steps, center, low, high)

    style_axis(ax, env_name, metric_name, all_plot_values, show_ylabel=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output_path = output_dir / f"{env_name}_{metric_name}_top4_by_diva_v1.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_combined_envs(env_grouped_runs, metric_name, smoothing, output_dir, show):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.figsize": [15, 4.8],
        }
    )

    env_names = list(env_grouped_runs.keys())
    fig, axes = plt.subplots(1, len(env_names), figsize=(5.2 * len(env_names), 4.8), squeeze=False)
    axes = axes[0]
    legend_handles = {}

    for idx, env_name in enumerate(env_names):
        ax = axes[idx]
        series, all_plot_values = compute_plot_series(env_grouped_runs[env_name], smoothing)
        for key, (steps, center, low, high) in series.items():
            plot_algorithm_series(ax, key, steps, center, low, high, label_override=DISPLAY_NAMES.get(key, key))
            if key not in legend_handles:
                legend_handles[key] = ax.lines[-1]
        style_axis(ax, env_name, metric_name, all_plot_values, show_ylabel=(idx == 0))

    ordered_handles = [legend_handles[key] for key in PLOT_ORDER if key in legend_handles]
    ordered_labels = [DISPLAY_NAMES.get(key, key) for key in PLOT_ORDER if key in legend_handles]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(len(ordered_labels), 5),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / f"combined_{metric_name}_top4_by_diva_v1.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    env_names = list(ENV_ROOTS) if args.env == "all" else [args.env]
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = (repo_root / args.output_dir).resolve()

    created = []
    env_grouped_runs = {}
    for env_name in env_names:
        root_dir = repo_root / ENV_ROOTS[env_name]
        runs = discover_runs(root_dir, args.metric, args.max_steps)
        selected_runs = select_canonical_runs(runs, env_name)
        grouped_runs = group_by_algorithm(selected_runs)
        env_grouped_runs[env_name] = grouped_runs

        print(f"{env_name}:")
        for algorithm in sorted(grouped_runs):
            print(f"  {algorithm}: {len(grouped_runs[algorithm])} run(s)")

        if not grouped_runs:
            print(f"  no runs found for metric {args.metric}")
            continue

        created.append(
            plot_env(env_name, grouped_runs, args.metric, args.smoothing, output_dir, args.show)
        )

    if args.combined and env_grouped_runs:
        created.append(
            plot_combined_envs(env_grouped_runs, args.metric, args.smoothing, output_dir, args.show)
        )

    if created:
        print("Created plots:")
        for path in created:
            print(f"- {path}")


if __name__ == "__main__":
    main()
