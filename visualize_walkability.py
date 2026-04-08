# visualize_walkability.py
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Basic utils
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def save_fig(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def safe_series(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def normalize_for_radar(values):
    arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.zeros_like(arr)
    valid = arr[mask]
    mn = valid.min()
    mx = valid.max()
    if abs(mx - mn) < 1e-12:
        out = np.zeros_like(arr)
        out[mask] = 1.0
        return out
    out = np.zeros_like(arr)
    out[mask] = (valid - mn) / (mx - mn)
    return out


def bin_center(df):
    return (df["bin_left"] + df["bin_right"]) / 2.0


def sort_metric(df, mean_col):
    s = pd.to_numeric(df[mean_col], errors="coerce")
    return df.assign(_sort=s).sort_values("_sort", ascending=False).drop(columns="_sort")


# =========================
# Plot functions
# =========================

def plot_scene_metric_bars(scene_df, output_dir):
    """
    One chart per metric, sorted descending.
    """
    metrics = [
        ("mean_heat_mean", "mean_heat_std", "Scene Mean Heat", "Mean heat"),
        ("continuity_index_mean", "continuity_index_std", "Scene Continuity Index", "Continuity index"),
        ("temporal_stability_mean", "temporal_stability_std", "Scene Temporal Stability", "Temporal stability"),
        ("road_gap_mean", "road_gap_std", "Scene Road Gap", "Road gap"),
        ("pedestrian_structure_mean", "pedestrian_structure_std", "Scene Pedestrian Structure", "Pedestrian structure"),
        ("dynamic_pressure_mean", "dynamic_pressure_std", "Scene Dynamic Pressure", "Dynamic pressure"),
    ]

    for mean_col, std_col, title, ylabel in metrics:
        if mean_col not in scene_df.columns:
            continue

        d = sort_metric(scene_df.copy(), mean_col)
        x = np.arange(len(d))
        y = pd.to_numeric(d[mean_col], errors="coerce").values
        yerr = pd.to_numeric(d[std_col], errors="coerce").values if std_col in d.columns else None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, y, yerr=yerr, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(d["scene_name"].astype(str).tolist(), rotation=30, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Scene")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        save_fig(fig, os.path.join(output_dir, f"{mean_col}_scene_bar.png"))


def plot_scene_heat_distribution(scene_hist_df, output_dir):
    """
    Compare scene-level heat distributions as line plots.
    """
    if scene_hist_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for scene_name, g in scene_hist_df.groupby("scene_name"):
        g = g.sort_values("bin_left")
        ax.plot(bin_center(g), g["ratio"], marker="o", label=str(scene_name))

    ax.set_title("Scene Heat Distribution")
    ax.set_xlabel("Mean heat bin center")
    ax.set_ylabel("Frame ratio")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    save_fig(fig, os.path.join(output_dir, "scene_heat_distribution_line.png"))


def plot_scene_heat_distribution_area(scene_hist_df, output_dir):
    """
    Stacked area-like representation using cumulative lines is less stable for many scenes.
    Here we create one line chart per scene as an alternative distribution display.
    """
    if scene_hist_df.empty:
        return

    ensure_dir(output_dir)

    for scene_name, g in scene_hist_df.groupby("scene_name"):
        g = g.sort_values("bin_left")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(bin_center(g), g["ratio"], alpha=0.35)
        ax.plot(bin_center(g), g["ratio"], marker="o")
        ax.set_title(f"Heat Distribution - {scene_name}")
        ax.set_xlabel("Mean heat bin center")
        ax.set_ylabel("Frame ratio")
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)

        safe_name = str(scene_name).replace("/", "_")
        save_fig(fig, os.path.join(output_dir, f"{safe_name}_heat_distribution_area.png"))


def plot_scene_continuity_scatter(scene_df, output_dir):
    """
    Continuity vs temporal stability: useful for 'walking continuity' dimension.
    """
    req = {"continuity_index_mean", "temporal_stability_mean", "scene_name"}
    if not req.issubset(scene_df.columns):
        return

    d = scene_df.copy()
    x = pd.to_numeric(d["continuity_index_mean"], errors="coerce").values
    y = pd.to_numeric(d["temporal_stability_mean"], errors="coerce").values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y)

    for _, row in d.iterrows():
        xi = pd.to_numeric(pd.Series([row["continuity_index_mean"]]), errors="coerce").iloc[0]
        yi = pd.to_numeric(pd.Series([row["temporal_stability_mean"]]), errors="coerce").iloc[0]
        ax.annotate(str(row["scene_name"]), (xi, yi), fontsize=9)

    ax.set_title("Scene Continuity vs Temporal Stability")
    ax.set_xlabel("Continuity index")
    ax.set_ylabel("Temporal stability")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_fig(fig, os.path.join(output_dir, "scene_continuity_vs_stability_scatter.png"))


def plot_scene_structure_scatter(scene_df, output_dir):
    """
    Pedestrian structure vs dynamic pressure: useful for 'street space feature' dimension.
    """
    req = {"pedestrian_structure_mean", "dynamic_pressure_mean", "scene_name"}
    if not req.issubset(scene_df.columns):
        return

    d = scene_df.copy()
    x = pd.to_numeric(d["pedestrian_structure_mean"], errors="coerce").values
    y = pd.to_numeric(d["dynamic_pressure_mean"], errors="coerce").values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y)

    for _, row in d.iterrows():
        xi = pd.to_numeric(pd.Series([row["pedestrian_structure_mean"]]), errors="coerce").iloc[0]
        yi = pd.to_numeric(pd.Series([row["dynamic_pressure_mean"]]), errors="coerce").iloc[0]
        ax.annotate(str(row["scene_name"]), (xi, yi), fontsize=9)

    ax.set_title("Scene Pedestrian Structure vs Dynamic Pressure")
    ax.set_xlabel("Pedestrian structure")
    ax.set_ylabel("Dynamic pressure")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_fig(fig, os.path.join(output_dir, "scene_structure_vs_dynamic_scatter.png"))


def plot_sequence_box_by_scene(seq_df, output_dir):
    """
    Show within-scene variation among sequences.
    One boxplot per metric.
    """
    metrics = [
        ("mean_heat_mean", "Sequence Mean Heat by Scene", "Mean heat"),
        ("continuity_index_mean", "Sequence Continuity by Scene", "Continuity index"),
        ("temporal_stability_mean", "Sequence Temporal Stability by Scene", "Temporal stability"),
        ("road_gap_mean", "Sequence Road Gap by Scene", "Road gap"),
        ("pedestrian_structure_mean", "Sequence Pedestrian Structure by Scene", "Pedestrian structure"),
        ("dynamic_pressure_mean", "Sequence Dynamic Pressure by Scene", "Dynamic pressure"),
    ]

    if seq_df.empty or "scene_name" not in seq_df.columns:
        return

    scene_order = sorted(seq_df["scene_name"].dropna().astype(str).unique().tolist())

    for col, title, ylabel in metrics:
        if col not in seq_df.columns:
            continue

        data = []
        labels = []
        for scene_name in scene_order:
            vals = pd.to_numeric(
                seq_df.loc[seq_df["scene_name"].astype(str) == scene_name, col],
                errors="coerce"
            ).dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(scene_name)

        if len(data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_title(title)
        ax.set_xlabel("Scene")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        save_fig(fig, os.path.join(output_dir, f"{col}_sequence_box_by_scene.png"))


def plot_sequence_heat_distribution(seq_hist_df, output_dir):
    """
    One figure per scene; each figure contains sequence-level heat distribution lines.
    """
    if seq_hist_df.empty:
        return

    ensure_dir(output_dir)

    for scene_name, gscene in seq_hist_df.groupby("scene_name"):
        fig, ax = plt.subplots(figsize=(10, 6))

        for seq_name, gseq in gscene.groupby("sequence_name"):
            gseq = gseq.sort_values("bin_left")
            ax.plot(bin_center(gseq), gseq["ratio"], marker="o", label=str(seq_name))

        ax.set_title(f"Sequence Heat Distribution - {scene_name}")
        ax.set_xlabel("Mean heat bin center")
        ax.set_ylabel("Frame ratio")
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

        safe_name = str(scene_name).replace("/", "_")
        save_fig(fig, os.path.join(output_dir, f"{safe_name}_sequence_heat_distribution.png"))


def plot_frame_trends(frame_df, output_dir, rolling_window=5):
    """
    For each sequence, create time-series line plots over frame index.
    """
    if frame_df.empty:
        return

    ensure_dir(output_dir)

    metrics = [
        ("mean_heat", "Frame Mean Heat", "Mean heat"),
        ("continuity_index", "Frame Continuity Index", "Continuity index"),
        ("temporal_stability", "Frame Temporal Stability", "Temporal stability"),
        ("road_gap", "Frame Road Gap", "Road gap"),
        ("pedestrian_structure", "Frame Pedestrian Structure", "Pedestrian structure"),
        ("dynamic_pressure", "Frame Dynamic Pressure", "Dynamic pressure"),
    ]

    group_cols = {"scene_name", "sequence_name", "frame_index"}
    if not group_cols.issubset(frame_df.columns):
        return

    for (scene_name, seq_name), g in frame_df.groupby(["scene_name", "sequence_name"]):
        g = g.copy()
        g["frame_index"] = pd.to_numeric(g["frame_index"], errors="coerce")
        g = g.sort_values("frame_index")

        seq_dir = os.path.join(output_dir, str(scene_name).replace("/", "_"))
        ensure_dir(seq_dir)

        for col, title, ylabel in metrics:
            if col not in g.columns:
                continue

            y = pd.to_numeric(g[col], errors="coerce")
            if y.notna().sum() == 0:
                continue

            smooth = y.rolling(window=rolling_window, min_periods=1).mean()

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(g["frame_index"], y, linewidth=1.0, label="Raw")
            ax.plot(g["frame_index"], smooth, linewidth=2.0, label=f"Rolling mean ({rolling_window})")
            ax.set_title(f"{title} - {scene_name} / {seq_name}")
            ax.set_xlabel("Frame index")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

            safe_seq = str(seq_name).replace("/", "_")
            save_fig(fig, os.path.join(seq_dir, f"{safe_seq}_{col}_trend.png"))


def plot_scene_radar(scene_df, output_dir):
    """
    Radar chart for six scene-level metrics.
    This is suitable as a compact summary figure.
    Values are min-max normalized across scenes.
    """
    required = [
        "scene_name",
        "mean_heat_mean",
        "continuity_index_mean",
        "temporal_stability_mean",
        "road_gap_mean",
        "pedestrian_structure_mean",
        "dynamic_pressure_mean",
    ]
    if not set(required).issubset(scene_df.columns):
        return

    d = scene_df[required].copy()

    # For radar, lower dynamic pressure is usually better.
    # Convert it to a positive orientation for display.
    d["dynamic_pressure_positive"] = 1.0 - pd.to_numeric(d["dynamic_pressure_mean"], errors="coerce")

    radar_cols = [
        "mean_heat_mean",
        "continuity_index_mean",
        "temporal_stability_mean",
        "road_gap_mean",
        "pedestrian_structure_mean",
        "dynamic_pressure_positive",
    ]
    radar_labels = [
        "Heat",
        "Continuity",
        "Stability",
        "Road gap",
        "Ped structure",
        "Low dynamic pressure",
    ]

    values_matrix = d[radar_cols].apply(pd.to_numeric, errors="coerce").values
    norm_matrix = np.zeros_like(values_matrix, dtype=float)

    for j in range(values_matrix.shape[1]):
        norm_matrix[:, j] = normalize_for_radar(values_matrix[:, j])

    n = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for i, scene_name in enumerate(d["scene_name"].astype(str).tolist()):
        vals = norm_matrix[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, label=scene_name)
        ax.fill(angles, vals, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_title("Scene Radar Summary (Normalized)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    save_fig(fig, os.path.join(output_dir, "scene_radar_summary.png"))


def plot_scene_metric_table(scene_df, output_dir):
    """
    Export a simple table figure for quick reporting.
    """
    required = [
        "scene_name",
        "mean_heat_mean",
        "continuity_index_mean",
        "temporal_stability_mean",
        "road_gap_mean",
        "pedestrian_structure_mean",
        "dynamic_pressure_mean",
    ]
    if not set(required).issubset(scene_df.columns):
        return

    d = scene_df[required].copy()
    d = d.round(3)

    fig, ax = plt.subplots(figsize=(12, 0.6 * (len(d) + 2)))
    ax.axis("off")

    table = ax.table(
        cellText=d.values,
        colLabels=d.columns,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    ax.set_title("Scene Metric Summary Table", pad=20)
    save_fig(fig, os.path.join(output_dir, "scene_metric_summary_table.png"))


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", type=str, required=True,
                        help="Directory containing summary CSV outputs.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all figures.")
    parser.add_argument("--rolling_window", type=int, default=5,
                        help="Rolling window for frame trend smoothing.")
    return parser.parse_args()


def main():
    args = parse_args()

    summary_dir = args.summary_dir
    output_dir = args.output_dir

    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "scene_level"))
    ensure_dir(os.path.join(output_dir, "sequence_level"))
    ensure_dir(os.path.join(output_dir, "frame_level"))

    frame_df = read_csv_if_exists(os.path.join(summary_dir, "all_frame_metrics.csv"))
    seq_df = read_csv_if_exists(os.path.join(summary_dir, "sequence_metrics.csv"))
    scene_df = read_csv_if_exists(os.path.join(summary_dir, "scene_metrics.csv"))
    seq_hist_df = read_csv_if_exists(os.path.join(summary_dir, "sequence_heat_distribution.csv"))
    scene_hist_df = read_csv_if_exists(os.path.join(summary_dir, "scene_heat_distribution.csv"))

    print("[INFO] Start plotting scene-level figures...")
    plot_scene_metric_bars(scene_df, os.path.join(output_dir, "scene_level"))
    plot_scene_heat_distribution(scene_hist_df, os.path.join(output_dir, "scene_level"))
    plot_scene_heat_distribution_area(scene_hist_df, os.path.join(output_dir, "scene_level", "scene_heat_single"))
    plot_scene_continuity_scatter(scene_df, os.path.join(output_dir, "scene_level"))
    plot_scene_structure_scatter(scene_df, os.path.join(output_dir, "scene_level"))
    plot_scene_radar(scene_df, os.path.join(output_dir, "scene_level"))
    plot_scene_metric_table(scene_df, os.path.join(output_dir, "scene_level"))

    print("[INFO] Start plotting sequence-level figures...")
    plot_sequence_box_by_scene(seq_df, os.path.join(output_dir, "sequence_level"))
    plot_sequence_heat_distribution(seq_hist_df, os.path.join(output_dir, "sequence_level", "sequence_heat"))

    print("[INFO] Start plotting frame-level figures...")
    plot_frame_trends(frame_df, os.path.join(output_dir, "frame_level"), rolling_window=args.rolling_window)

    print("[INFO] Done.")
    print(f"[INFO] Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

    # python visualize_walkability.py --summary_dir ./summary_outputs --output_dir ./summary_figures --rolling_window 9
