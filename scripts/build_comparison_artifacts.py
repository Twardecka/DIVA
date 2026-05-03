#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


SOURCES = (
    ("baseline_old", Path("results_basleines/sacred")),
    ("baseline_new", Path("results/sacred")),
    ("diva_v1", Path("results_DIVA softplus/sacred")),
    ("diva_v2", Path("results diva changed (2)/sacred")),
)

TARGET_ENVS = ("gather", "hallway", "disperse")

METRICS_TO_EXPORT = (
    "test_battle_won_mean",
    "test_return_mean",
    "test_win_group_mean",
    "test_match_mean",
    "test_ep_length_mean",
    "battle_won_mean",
    "return_mean",
)

ALGORITHM_LABELS = {
    "diva_bounded_sigmoid_qmix_DIVA": "diva_v1",
    "diva_bounded_sigmoid_qmix_DIVA_scale1_capacity64_gate2": "diva_v2",
    "diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5": "diva_v3",
}

RESULT_FOLDER_NAMES = {
    "gather": "results_all_gather",
    "hallway": "results_all_hallway",
    "disperse": "results_all_disperse",
}

RESULT_BUNDLE_SOURCE_PREFERENCE = {
    "gather": {
        "qmix": "baseline_new",
        "vdn": "baseline_new",
    },
    "hallway": {
        "qmix": "baseline_old",
        "vdn": "baseline_old",
        "qtran": "baseline_old",
    },
    "disperse": {
        "qmix": "baseline_old",
        "vdn": "baseline_old",
        "qtran": "baseline_old",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-environment result tables and clean TensorBoard exports from Sacred runs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/comparison_results"),
        help="Directory where CSV summaries and TensorBoard exports will be written.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete previously generated comparison artifacts before writing new ones.",
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
            scalar = scalar_to_float(value)
            if scalar is None:
                continue
            try:
                step_int = int(step)
            except (TypeError, ValueError):
                continue
            pairs.append((step_int, scalar))
        if pairs:
            yield key, pairs


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def algorithm_label(raw_name):
    return ALGORITHM_LABELS.get(raw_name, raw_name)


def classify_run(source_name, run_id, raw_name):
    if source_name == "baseline_new":
        if raw_name in {"qmix", "vdn", "qtran"}:
            return source_name, algorithm_label(raw_name)
        if raw_name == "diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5" and 13 <= run_id <= 20:
            return "diva_v3", algorithm_label(raw_name)
        return None, None
    return source_name, algorithm_label(raw_name)


def sanitize_component(value):
    safe = str(value).strip().replace("/", "_").replace(" ", "_")
    return safe or "unknown"


def safe_rmtree(path):
    path = Path(path)
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        return
    except OSError:
        pass

    for root, dirnames, filenames in os.walk(path, topdown=False):
        root_path = Path(root)
        for filename in filenames:
            try:
                (root_path / filename).unlink()
            except FileNotFoundError:
                pass
        for dirname in dirnames:
            try:
                (root_path / dirname).rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        shutil.rmtree(path, ignore_errors=True)


def gather_runs(repo_root):
    rows = []
    duplicate_counter = Counter()

    for source_name, rel_root in SOURCES:
        sacred_root = repo_root / rel_root
        if not sacred_root.exists():
            continue

        for run_dir in sorted(
            (path for path in sacred_root.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        ):
            config_path = run_dir / "config.json"
            info_path = run_dir / "info.json"
            run_path = run_dir / "run.json"

            if not (config_path.exists() and info_path.exists() and run_path.exists()):
                continue

            try:
                config = read_json(config_path)
                info = read_json(info_path)
                run_data = read_json(run_path)
            except Exception:
                continue

            env_name = config.get("env")
            if env_name not in TARGET_ENVS:
                continue

            raw_name = config.get("name", "unknown_alg")
            seed = config.get("seed", "unknown_seed")
            run_id = int(run_dir.name)
            classified_source, alg_label = classify_run(source_name, run_id, raw_name)
            if classified_source is None:
                continue
            latest_step = None
            series_by_key = {}
            for key, pairs in metric_series(info):
                series_by_key[key] = pairs
                step = pairs[-1][0]
                latest_step = step if latest_step is None else max(latest_step, step)

            row = {
                "source_group": classified_source,
                "sacred_root": str(rel_root),
                "run_id": run_id,
                "run_dir": str(run_dir.relative_to(repo_root)),
                "run_dir_abs": str(run_dir),
                "env": env_name,
                "algorithm": alg_label,
                "raw_algorithm": raw_name,
                "seed": seed,
                "status": run_data.get("status", "UNKNOWN"),
                "start_time": run_data.get("start_time"),
                "latest_step": latest_step,
                "config": config,
                "run_data": run_data,
                "info": info,
                "series_by_key": series_by_key,
            }

            for metric_name in METRICS_TO_EXPORT:
                pairs = series_by_key.get(metric_name, [])
                last_value = pairs[-1][1] if pairs else None
                best_value = max((value for _, value in pairs), default=None)
                row[f"{metric_name}_last"] = last_value
                row[f"{metric_name}_best"] = best_value

            duplicate_counter[(env_name, alg_label, seed)] += 1
            rows.append(row)

    for row in rows:
        key = (row["env"], row["algorithm"], row["seed"])
        base_label = f"{row['algorithm']}-seed{row['seed']}"
        if duplicate_counter[key] == 1:
            tb_name = base_label
        else:
            tb_name = f"{base_label}-{row['source_group']}-run{row['run_id']}"
        row["tb_name"] = sanitize_component(tb_name)

    return rows


def csv_fieldnames():
    fieldnames = [
        "env",
        "algorithm",
        "raw_algorithm",
        "seed",
        "source_group",
        "run_id",
        "status",
        "start_time",
        "latest_step",
        "tb_name",
        "run_dir",
        "sacred_root",
    ]
    for metric_name in METRICS_TO_EXPORT:
        fieldnames.append(f"{metric_name}_last")
        fieldnames.append(f"{metric_name}_best")
    return fieldnames


def metric_or_neg_inf(row, key):
    value = row.get(key)
    if value is None:
        return float("-inf")
    return value


def select_top4_diva_v1_seeds(rows, env_name):
    return select_top4_diva_seeds(rows, env_name, "diva_v1", "diva_v1")


def select_top4_diva_v2_seeds(rows, env_name):
    return select_top4_diva_seeds(rows, env_name, "diva_v2", "diva_v2")


def select_top4_diva_v3_seeds(rows, env_name):
    return select_top4_diva_seeds(rows, env_name, "diva_v3", "diva_v3")


def select_top4_diva_seeds(rows, env_name, algorithm, source_group):
    diva_rows = [
        row
        for row in rows
        if row["env"] == env_name and row["algorithm"] == algorithm and row["source_group"] == source_group
    ]

    def secondary_progress(row):
        for key in ("test_win_group_mean_best", "test_match_mean_best", "test_return_mean_best"):
            value = row.get(key)
            if value is not None:
                return value
        return float("-inf")

    ranked_rows = sorted(
        diva_rows,
        key=lambda row: (
            -metric_or_neg_inf(row, "test_battle_won_mean_best"),
            -metric_or_neg_inf(row, "test_return_mean_best"),
            -secondary_progress(row),
            int(row["seed"]),
        ),
    )
    return [int(row["seed"]) for row in ranked_rows[:4]]


def write_csvs(rows, csv_dir):
    csv_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames()

    for env_name in TARGET_ENVS:
        env_rows = [row for row in rows if row["env"] == env_name]
        env_rows.sort(key=lambda row: (row["algorithm"], row["seed"], row["source_group"], row["run_id"]))
        output_path = csv_dir / f"{env_name}_results.csv"
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in env_rows:
                writer.writerow({key: row.get(key) for key in fieldnames})


def write_tensorboard(rows, tb_dir, group_by_env):
    tb_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        run_tb_dir = tb_dir
        if group_by_env:
            run_tb_dir = run_tb_dir / sanitize_component(row["env"])
        run_tb_dir = run_tb_dir / row["tb_name"]
        run_tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_tb_dir))

        for key, pairs in row["series_by_key"].items():
            for step, value in pairs:
                writer.add_scalar(key, value, step)

        metadata = {
            "source_group": row["source_group"],
            "run_id": row["run_id"],
            "run_dir": row["run_dir"],
            "algorithm": row["algorithm"],
            "raw_algorithm": row["raw_algorithm"],
            "env": row["env"],
            "seed": row["seed"],
            "status": row["status"],
            "start_time": row["start_time"],
            "latest_step": row["latest_step"],
        }
        writer.add_text("run/config", json.dumps(row["config"], indent=2), 0)
        writer.add_text("run/metadata", json.dumps(metadata, indent=2), 0)
        writer.flush()
        writer.close()


def write_result_bundle(bundle_dir, env_name, env_rows, fieldnames, csv_filename, readme_lines):
    sacred_dir = bundle_dir / "sacred"
    tb_dir = bundle_dir / "tb_logs"
    sacred_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    for row in env_rows:
        src_dir = Path(row["run_dir_abs"])
        dst_dir = sacred_dir / row["tb_name"]
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

    manifest_path = bundle_dir / csv_filename
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in env_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    write_tensorboard(env_rows, tb_dir, group_by_env=False)

    readme_path = bundle_dir / "README.txt"
    with readme_path.open("w") as handle:
        handle.write("\n".join(readme_lines) + "\n")


def filter_result_bundle_rows(env_rows, env_name):
    preferences = RESULT_BUNDLE_SOURCE_PREFERENCE.get(env_name, {})
    filtered_rows = []
    for row in env_rows:
        preferred_source = preferences.get(row["algorithm"])
        if preferred_source is not None and row["source_group"] != preferred_source:
            continue
        filtered_rows.append(row)
    return filtered_rows


def write_result_style_dirs(rows, repo_root):
    fieldnames = csv_fieldnames()
    top4_seed_map = {}

    for env_name in TARGET_ENVS:
        env_rows = [row for row in rows if row["env"] == env_name]
        env_rows.sort(key=lambda row: (row["algorithm"], row["seed"], row["source_group"], row["run_id"]))
        bundle_rows = filter_result_bundle_rows(env_rows, env_name)
        top4_v1_seeds = select_top4_diva_v1_seeds(rows, env_name)
        top4_v2_seeds = select_top4_diva_v2_seeds(rows, env_name)
        top4_v3_seeds = select_top4_diva_v3_seeds(rows, env_name)
        top4_seed_map[env_name] = {"diva_v1": top4_v1_seeds, "diva_v2": top4_v2_seeds, "diva_v3": top4_v3_seeds}

        results_dir = repo_root / RESULT_FOLDER_NAMES[env_name]
        write_result_bundle(
            results_dir,
            env_name,
            bundle_rows,
            fieldnames,
            f"{env_name}_results.csv",
            [
                RESULT_FOLDER_NAMES[env_name],
                "=" * len(RESULT_FOLDER_NAMES[env_name]),
                "",
                "Contents:",
                "- sacred/<algorithm-seed...>: copied Sacred run folders for this environment.",
                "- tb_logs/<algorithm-seed...>: clean TensorBoard logs for this environment.",
                f"- {env_name}_results.csv: flat manifest of all runs in this folder.",
                "- top4_by_diva_v1/: only the top 4 diva_v1-ranked seeds for this environment.",
                "",
                "Algorithm labels:",
                "- diva_v1 = diva_bounded_sigmoid_qmix_DIVA",
                "- diva_v2 = diva_bounded_sigmoid_qmix_DIVA_scale1_capacity64_gate2",
                "- diva_v3 = results/sacred/13-20 (diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5 runs)",
                "",
                f"Top 4 diva_v1 seeds: {', '.join(str(seed) for seed in top4_v1_seeds)}",
                f"Top 4 diva_v2 seeds: {', '.join(str(seed) for seed in top4_v2_seeds)}",
                f"Top 4 diva_v3 seeds: {', '.join(str(seed) for seed in top4_v3_seeds)}",
            ],
        )

        top4_dir = results_dir / "top4_by_diva_v1"
        top4_rows = [
            row
            for row in bundle_rows
            if (
                (row["algorithm"] == "diva_v1" and int(row["seed"]) in top4_v1_seeds)
                or (row["algorithm"] == "diva_v2" and int(row["seed"]) in top4_v2_seeds)
                or (row["algorithm"] == "diva_v3" and int(row["seed"]) in top4_v3_seeds)
                or (row["algorithm"] not in {"diva_v1", "diva_v2", "diva_v3"} and int(row["seed"]) in top4_v1_seeds)
            )
        ]
        write_result_bundle(
            top4_dir,
            env_name,
            top4_rows,
            fieldnames,
            f"{env_name}_top4_by_diva_v1_results.csv",
            [
                f"{RESULT_FOLDER_NAMES[env_name]} top4_by_diva_v1",
                "=" * (len(RESULT_FOLDER_NAMES[env_name]) + len(" top4_by_diva_v1")),
                "",
                f"Selected seeds from diva_v1 for baselines/diva_v1: {', '.join(str(seed) for seed in top4_v1_seeds)}",
                f"Selected seeds from diva_v2 for diva_v2: {', '.join(str(seed) for seed in top4_v2_seeds)}",
                f"Selected seeds from diva_v3 for diva_v3: {', '.join(str(seed) for seed in top4_v3_seeds)}",
                "Ranking rule:",
                "- primary: test_battle_won_mean_best",
                "- tie-break 1: test_return_mean_best",
                "- tie-break 2: env progress metric when available (win_group or match)",
                "",
                "Contents:",
                "- sacred/<algorithm-seed...>: copied Sacred run folders for the selected seeds only.",
                "- tb_logs/<algorithm-seed...>: clean TensorBoard logs for the selected seeds only.",
                f"- {env_name}_top4_by_diva_v1_results.csv: flat manifest of the selected runs.",
            ],
        )

    return top4_seed_map


def write_readme(rows, output_dir):
    counts = defaultdict(int)
    for row in rows:
        counts[row["env"]] += 1

    readme_path = output_dir / "README.txt"
    with readme_path.open("w") as handle:
        handle.write("comparison_results\n")
        handle.write("==================\n\n")
        handle.write("Generated files:\n")
        handle.write("- csv/<env>_results.csv: one row per Sacred run for gather, hallway, and disperse.\n")
        handle.write("- tb_logs/<env>/<algorithm>-seed<seed>[...]: TensorBoard export with clean per-run names.\n\n")
        handle.write("Algorithm labels:\n")
        handle.write("- diva_v1 = diva_bounded_sigmoid_qmix_DIVA\n")
        handle.write("- diva_v2 = diva_bounded_sigmoid_qmix_DIVA_scale1_capacity64_gate2\n\n")
        handle.write("- diva_v3 = results/sacred/13-20 (diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5)\n\n")
        handle.write("Run counts by environment:\n")
        for env_name in TARGET_ENVS:
            handle.write(f"- {env_name}: {counts.get(env_name, 0)} runs\n")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = (repo_root / args.output_dir).resolve()

    if args.purge and output_dir.exists():
        safe_rmtree(output_dir)
    if args.purge:
        for env_name in TARGET_ENVS:
            results_dir = repo_root / RESULT_FOLDER_NAMES[env_name]
            if results_dir.exists():
                safe_rmtree(results_dir)

    rows = gather_runs(repo_root)
    write_csvs(rows, output_dir / "csv")
    write_tensorboard(rows, output_dir / "tb_logs", group_by_env=True)
    write_readme(rows, output_dir)
    top4_seed_map = write_result_style_dirs(rows, repo_root)

    print(f"Wrote comparison artifacts to {output_dir}")
    print("Created result-style folders:")
    for env_name in TARGET_ENVS:
        print(f"- {repo_root / RESULT_FOLDER_NAMES[env_name]}")
        print(f"  top4 diva_v1 seeds: {top4_seed_map[env_name]['diva_v1']}")
        print(f"  top4 diva_v2 seeds: {top4_seed_map[env_name]['diva_v2']}")
        print(f"  top4 diva_v3 seeds: {top4_seed_map[env_name]['diva_v3']}")
    for env_name in TARGET_ENVS:
        env_count = sum(1 for row in rows if row['env'] == env_name)
        print(f"{env_name}: {env_count} runs")


if __name__ == "__main__":
    main()
