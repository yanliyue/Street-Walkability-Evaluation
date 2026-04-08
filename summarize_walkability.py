# summarize_walkability.py
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from glob import glob

import numpy as np
import pandas as pd


EPS = 1e-6


# =========================
# Basic utils
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_subdirs(path: str):
    if not os.path.isdir(path):
        return []
    dirs = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and not name.startswith("."):
            dirs.append(name)
    return sorted(dirs)


def json_sort_key(fp: str):
    name = os.path.splitext(os.path.basename(fp))[0]
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def safe_float(x, default=np.nan):
    if x is None:
        return default
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def nan_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    q25 = float(np.quantile(arr, 0.25))
    q75 = float(np.quantile(arr, 0.75))
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "q25": q25,
        "q75": q75,
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def add_stats_to_record(record: dict, prefix: str, values):
    stats = nan_stats(values)
    for k, v in stats.items():
        record[f"{prefix}_{k}"] = v


def build_hist_records(values, scene_name, sequence_name=None, bins=10):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return []

    hist, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    total = int(hist.sum())
    rows = []
    for i in range(len(hist)):
        rows.append({
            "scene_name": scene_name,
            "sequence_name": sequence_name,
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "count": int(hist[i]),
            "ratio": float(hist[i] / total) if total > 0 else np.nan,
        })
    return rows


# =========================
# Metric calculation
# =========================

def compute_frame_metrics_from_json(data: dict):
    analysis = data.get("analysis", {})
    temporal = data.get("temporal", {})

    mean_heat = safe_float(analysis.get("mean_stable_heat"))
    reachable_ratio = safe_float(analysis.get("reachable_ratio"), 0.0)
    high_walk_ratio = safe_float(analysis.get("high_walk_ratio"), 0.0)
    mid_walk_ratio = safe_float(analysis.get("mid_walk_ratio"), 0.0)
    dynamic_ratio = safe_float(analysis.get("dynamic_ratio"), 0.0)

    heat_on_high_walk = safe_float(analysis.get("heat_on_high_walk"))
    heat_on_mid_walk = safe_float(analysis.get("heat_on_mid_walk"))

    temporal_stability = safe_float(temporal.get("stability_gain"))

    # 1) 步行连续性
    continuity_index = (
        reachable_ratio
        * (0.7 * high_walk_ratio + 0.3 * mid_walk_ratio)
        * (1.0 - dynamic_ratio)
    )

    # 2) 道路内部差异
    if np.isnan(heat_on_high_walk) or np.isnan(heat_on_mid_walk):
        road_gap = np.nan
    else:
        road_gap = float(heat_on_high_walk - heat_on_mid_walk)

    # 3) 街道空间特征
    pedestrian_structure = float(
        high_walk_ratio / (high_walk_ratio + mid_walk_ratio + EPS)
    )
    dynamic_pressure = float(dynamic_ratio)

    return {
        "mean_heat": mean_heat,                       # 热度水平
        "continuity_index": continuity_index,        # 步行连续性
        "temporal_stability": temporal_stability,    # 时序稳定性
        "road_gap": road_gap,                        # 道路内部差异
        "pedestrian_structure": pedestrian_structure,# 步行结构主导度
        "dynamic_pressure": dynamic_pressure,        # 动态压力
        # 保留基础字段，方便追溯
        "reachable_ratio": reachable_ratio,
        "high_walk_ratio": high_walk_ratio,
        "mid_walk_ratio": mid_walk_ratio,
        "dynamic_ratio": dynamic_ratio,
        "heat_on_high_walk": heat_on_high_walk,
        "heat_on_mid_walk": heat_on_mid_walk,
    }


# =========================
# Data scan
# =========================

def scan_all_frames(root_dir: str):
    frame_rows = []
    skipped_rows = []

    scene_names = list_subdirs(root_dir)

    for scene_name in scene_names:
        scene_dir = os.path.join(root_dir, scene_name)
        sequence_names = list_subdirs(scene_dir)

        if len(sequence_names) == 0:
            skipped_rows.append({
                "scene_name": scene_name,
                "sequence_name": None,
                "reason": "scene_has_no_sequence_subdir",
                "path": scene_dir,
            })
            continue

        for sequence_name in sequence_names:
            seq_dir = os.path.join(scene_dir, sequence_name)
            explain_dir = os.path.join(seq_dir, "explain")

            if not os.path.isdir(explain_dir):
                skipped_rows.append({
                    "scene_name": scene_name,
                    "sequence_name": sequence_name,
                    "reason": "missing_explain_dir",
                    "path": explain_dir,
                })
                continue

            json_files = sorted(glob(os.path.join(explain_dir, "*.json")), key=json_sort_key)

            if len(json_files) == 0:
                skipped_rows.append({
                    "scene_name": scene_name,
                    "sequence_name": sequence_name,
                    "reason": "empty_explain_dir",
                    "path": explain_dir,
                })
                continue

            for fp in json_files:
                frame_file = os.path.basename(fp)
                frame_name = os.path.splitext(frame_file)[0]
                frame_index = int(frame_name) if frame_name.isdigit() else np.nan

                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    metric_row = compute_frame_metrics_from_json(data)

                    row = {
                        "scene_name": scene_name,
                        "sequence_name": sequence_name,
                        "frame_name": frame_name,
                        "frame_index": frame_index,
                        "json_path": fp,
                    }
                    row.update(metric_row)
                    frame_rows.append(row)

                except Exception as e:
                    skipped_rows.append({
                        "scene_name": scene_name,
                        "sequence_name": sequence_name,
                        "reason": f"json_read_or_parse_error: {str(e)}",
                        "path": fp,
                    })

    df_frames = pd.DataFrame(frame_rows)
    df_skipped = pd.DataFrame(skipped_rows)
    return df_frames, df_skipped


# =========================
# Sequence aggregation
# =========================

def summarize_sequences(df_frames: pd.DataFrame):
    metric_cols = [
        "mean_heat",
        "continuity_index",
        "temporal_stability",
        "road_gap",
        "pedestrian_structure",
        "dynamic_pressure",
    ]

    rows = []
    hist_rows = []

    if df_frames.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = df_frames.groupby(["scene_name", "sequence_name"], dropna=False)

    for (scene_name, sequence_name), g in grouped:
        g = g.sort_values("frame_index", na_position="last")

        record = {
            "scene_name": scene_name,
            "sequence_name": sequence_name,
            "n_frames": int(len(g)),
        }

        for col in metric_cols:
            add_stats_to_record(record, col, g[col].values)

        rows.append(record)

        # 热度分布：按帧 mean_heat 做直方分布
        hist_rows.extend(
            build_hist_records(
                g["mean_heat"].values,
                scene_name=scene_name,
                sequence_name=sequence_name,
                bins=10,
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(hist_rows)


# =========================
# Scene aggregation
# =========================

def summarize_scenes(df_frames: pd.DataFrame, df_seq: pd.DataFrame):
    """
    场景级默认按“视频序列等权”汇总：
    即对每个视频序列的 metric_mean 再做场景统计。
    这样不会让长视频因帧数多而主导整个场景结果。
    """
    metric_cols = [
        "mean_heat",
        "continuity_index",
        "temporal_stability",
        "road_gap",
        "pedestrian_structure",
        "dynamic_pressure",
    ]

    scene_rows = []
    scene_hist_rows = []

    if df_seq.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped_scene_seq = df_seq.groupby("scene_name", dropna=False)

    for scene_name, gseq in grouped_scene_seq:
        record = {
            "scene_name": scene_name,
            "n_sequences": int(len(gseq)),
            "total_frames": int(
                gseq["n_frames"].sum() if "n_frames" in gseq.columns else 0
            ),
        }

        for col in metric_cols:
            seq_level_values = gseq[f"{col}_mean"].values
            add_stats_to_record(record, col, seq_level_values)

        scene_rows.append(record)

    # 场景热度分布：按该场景所有帧的 mean_heat 生成
    grouped_scene_frame = df_frames.groupby("scene_name", dropna=False)
    for scene_name, gframe in grouped_scene_frame:
        scene_hist_rows.extend(
            build_hist_records(
                gframe["mean_heat"].values,
                scene_name=scene_name,
                sequence_name=None,
                bins=10,
            )
        )

    return pd.DataFrame(scene_rows), pd.DataFrame(scene_hist_rows)


# =========================
# JSON export
# =========================

def dataframe_to_json_records(df: pd.DataFrame, save_path: str):
    if df.empty:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    records = df.replace({np.nan: None}).to_dict(orient="records")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="场景总目录：其下一级为场景文件夹，场景下一级为视频序列文件夹")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="汇总结果保存目录")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print("[INFO] Scanning frame-level explain json files...")
    df_frames, df_skipped = scan_all_frames(args.root_dir)

    frame_csv = os.path.join(args.output_dir, "all_frame_metrics.csv")
    skipped_csv = os.path.join(args.output_dir, "skipped_items.csv")

    df_frames.to_csv(frame_csv, index=False, encoding="utf-8-sig")
    df_skipped.to_csv(skipped_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] Frame metrics saved: {frame_csv}")
    print(f"[INFO] Skipped log saved:   {skipped_csv}")

    print("[INFO] Aggregating sequence-level metrics...")
    df_seq, df_seq_hist = summarize_sequences(df_frames)

    seq_csv = os.path.join(args.output_dir, "sequence_metrics.csv")
    seq_hist_csv = os.path.join(args.output_dir, "sequence_heat_distribution.csv")
    df_seq.to_csv(seq_csv, index=False, encoding="utf-8-sig")
    df_seq_hist.to_csv(seq_hist_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] Sequence metrics saved:        {seq_csv}")
    print(f"[INFO] Sequence heat distribution:   {seq_hist_csv}")

    print("[INFO] Aggregating scene-level metrics...")
    df_scene, df_scene_hist = summarize_scenes(df_frames, df_seq)

    scene_csv = os.path.join(args.output_dir, "scene_metrics.csv")
    scene_hist_csv = os.path.join(args.output_dir, "scene_heat_distribution.csv")
    df_scene.to_csv(scene_csv, index=False, encoding="utf-8-sig")
    df_scene_hist.to_csv(scene_hist_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] Scene metrics saved:           {scene_csv}")
    print(f"[INFO] Scene heat distribution:      {scene_hist_csv}")

    # 同时导出 JSON，方便后续程序直接读取
    dataframe_to_json_records(df_frames, os.path.join(args.output_dir, "all_frame_metrics.json"))
    dataframe_to_json_records(df_seq, os.path.join(args.output_dir, "sequence_metrics.json"))
    dataframe_to_json_records(df_scene, os.path.join(args.output_dir, "scene_metrics.json"))
    dataframe_to_json_records(df_seq_hist, os.path.join(args.output_dir, "sequence_heat_distribution.json"))
    dataframe_to_json_records(df_scene_hist, os.path.join(args.output_dir, "scene_heat_distribution.json"))
    dataframe_to_json_records(df_skipped, os.path.join(args.output_dir, "skipped_items.json"))

    print("[INFO] Done.")
    print(f"[INFO] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


    # python summarize_walkability.py --root_dir ./outputs --output_dir ./summary_outputs
