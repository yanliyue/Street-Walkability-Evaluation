# run_demo.py
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from glob import glob
from collections import deque

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
)


# =========================
# Basic utils
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sorted_image_list(input_dir: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(input_dir, ext)))
    return sorted(files)


def resize_keep_aspect(bgr: np.ndarray, long_side: int):
    if long_side <= 0:
        return bgr
    h, w = bgr.shape[:2]
    cur_long = max(h, w)
    if cur_long <= long_side:
        return bgr
    scale = long_side / float(cur_long)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out


def normalize_01(x: np.ndarray, eps: float = 1e-6):
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def deterministic_color(idx: int):
    # Stable pseudo-random color per label id
    r = (37 * idx + 73) % 255
    g = (17 * idx + 131) % 255
    b = (29 * idx + 191) % 255
    return (b, g, r)


def label_map_to_color(label_map: np.ndarray):
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_map)
    for idx in ids:
        color[label_map == idx] = deterministic_color(int(idx))
    return color


def save_heatmap_png(heatmap: np.ndarray, save_path: str):
    x = np.clip(heatmap, 0.0, 1.0)
    x_u8 = (x * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(x_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(save_path, vis)
    return vis


def overlay_heatmap_on_bgr(bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42):
    x = np.clip(heatmap, 0.0, 1.0)
    x_u8 = (x * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(x_u8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(bgr, 1.0 - alpha, heat, alpha, 0)
    return overlay


# =========================
# Model loading
# =========================

def load_segmentation_model(model_dir: str, device: str):
    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return processor, model


def load_depth_model(model_dir: str, device: str):
    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return processor, model


# =========================
# Inference
# =========================

@torch.inference_mode()
def predict_semantic(rgb: np.ndarray, processor, model, device: str):
    """
    Return:
        label_map: H x W int32
        id2label: dict[int, str]
        color_seg: H x W x 3 BGR
    """
    pil = Image.fromarray(rgb)
    h, w = rgb.shape[:2]

    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    seg = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(h, w)]
    )[0]

    label_map = seg.detach().cpu().numpy().astype(np.int32)

    raw_id2label = model.config.id2label
    id2label = {}
    for k, v in raw_id2label.items():
        try:
            id2label[int(k)] = v
        except Exception:
            id2label[k] = v

    color_rgb = label_map_to_color(label_map)
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    return label_map, id2label, color_bgr


@torch.inference_mode()
def predict_depth(rgb: np.ndarray, processor, model, device: str, depth_mode: str):
    """
    depth_mode:
        - near_is_high: predicted depth visualization is near-bright / far-dark,
                        heatmap uses openness = 1 - normalized_depth
        - far_is_high : predicted depth visualization is far-bright / near-dark,
                        heatmap uses openness = normalized_depth

    Return:
        depth_open_map: H x W float32 in [0,1], larger means more open / farther
        depth_vis: H x W uint8 BGR for saving
        depth_norm_vis_source: H x W float32 raw normalized depth for inspection
    """
    pil = Image.fromarray(rgb)
    h, w = rgb.shape[:2]

    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # [B, H, W] usually

    if predicted_depth.ndim == 3:
        predicted_depth = predicted_depth.unsqueeze(1)  # [B,1,H,W]

    depth = F.interpolate(
        predicted_depth,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )

    depth = depth[0, 0].detach().cpu().numpy().astype(np.float32)
    depth_norm = normalize_01(depth)

    if depth_mode == "near_is_high":
        depth_open = 1.0 - depth_norm
        depth_vis_gray = (depth_norm * 255.0).astype(np.uint8)
    else:
        depth_open = depth_norm
        depth_vis_gray = (depth_norm * 255.0).astype(np.uint8)

    depth_vis = cv2.cvtColor(depth_vis_gray, cv2.COLOR_GRAY2BGR)
    return depth_open.astype(np.float32), depth_vis, depth_norm.astype(np.float32)


# =========================
# Semantic grouping
# =========================

def normalize_label_name(name: str):
    return name.lower().replace("-", " ").replace("_", " ").strip()


def find_label_ids(id2label):
    groups = {
        "high_walk": [],
        "mid_walk": [],
        "obstacle": [],
        "vehicle": [],
        "light_dynamic": [],
        "sky": [],
        "building": [],
    }

    for idx, label in id2label.items():
        name = normalize_label_name(str(label))

        if any(k in name for k in [
            "sidewalk",
            "crosswalk",
            "pedestrian area",
            "lane marking crosswalk",
            "zebra"
        ]):
            groups["high_walk"].append(int(idx))
            continue

        if any(k in name for k in [
            "road",
            "service lane",
            "bike lane",
            "street",
            "driveway",
            "alley",
            "residential road"
        ]):
            groups["mid_walk"].append(int(idx))
            continue

        if any(k in name for k in [
            "car",
            "bus",
            "truck",
            "vehicle",
            "van",
            "motorcycle"
        ]):
            groups["vehicle"].append(int(idx))
            continue

        if any(k in name for k in [
            "person",
            "pedestrian",
            "bicyclist",
            "cyclist",
            "bicycle",
            "rider"
        ]):
            groups["light_dynamic"].append(int(idx))
            continue

        if "sky" in name:
            groups["sky"].append(int(idx))
            continue

        if any(k in name for k in [
            "building",
            "house",
            "garage",
            "bridge",
            "tunnel"
        ]):
            groups["building"].append(int(idx))
            continue

        if any(k in name for k in [
            "wall",
            "fence",
            "pole",
            "traffic sign",
            "traffic light",
            "curb",
            "barrier",
            "rail",
            "guard rail",
            "sign",
            "lamp",
            "bench",
            "trash",
            "vegetation",
            "terrain"
        ]):
            groups["obstacle"].append(int(idx))
            continue

    return groups

def horizontal_run_width_map(mask: np.ndarray):
    """
    对 mask 中每一行的连续区域，赋值其横向宽度。
    适合前视街景估计“路有多宽”。
    """
    h, w = mask.shape
    width_map = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        x = 0
        while x < w:
            if mask[y, x] == 0:
                x += 1
                continue
            x0 = x
            while x < w and mask[y, x] > 0:
                x += 1
            x1 = x - 1
            run_w = float(x1 - x0 + 1)
            width_map[y, x0:x1 + 1] = run_w

    width_map = cv2.GaussianBlur(width_map, (17, 17), 0)
    return width_map

def refine_mid_walk_score(
    mid_walk: np.ndarray,
    reachable_ground: np.ndarray,
    obstacle_mask: np.ndarray,
    building_mask: np.ndarray,
    vehicle_mask: np.ndarray,
    light_dynamic_mask: np.ndarray,
    depth_open_map: np.ndarray,
):
    """
    对 road-like 区域做二次打分：
    窄巷 / 两侧建筑夹持 / 车少 -> 提级
    宽路 / 开阔 / 车辆多 -> 降级
    """
    h, w = mid_walk.shape

    road_mask = ((mid_walk > 0.5) & (reachable_ground > 0.5)).astype(np.uint8)
    if road_mask.sum() == 0:
        return np.zeros_like(depth_open_map, dtype=np.float32)

    # 边界：建筑 + 静态障碍
    boundary_mask = ((obstacle_mask > 0.5) | (building_mask > 0.5)).astype(np.uint8)

    # 1) 横向局部宽度
    road_width_px = horizontal_run_width_map(road_mask)
    road_width_norm = road_width_px / float(w + 1e-6)

    # 窄路 bonus：<= 0.12w 很强，>= 0.30w 基本无 bonus
    narrow_bonus = 1.0 - np.clip((road_width_norm - 0.12) / 0.18, 0.0, 1.0)

    # 宽路 penalty：>= 0.30w 开始惩罚
    wide_penalty = np.clip((road_width_norm - 0.30) / 0.20, 0.0, 1.0)

    # 2) 边界接近程度：越靠近 building/wall/curb，越像巷道
    boundary_dist = cv2.distanceTransform(1 - boundary_mask, cv2.DIST_L2, 5)
    boundary_near = 1.0 - np.clip(boundary_dist / (0.08 * w + 1e-6), 0.0, 1.0)

    # 3) 车辆密度
    vehicle_density = cv2.GaussianBlur(vehicle_mask.astype(np.float32), (21, 21), 0)
    light_dynamic_density = cv2.GaussianBlur(light_dynamic_mask.astype(np.float32), (15, 15), 0)

    # 4) 开阔度惩罚：特别开阔更像主路
    openness_penalty = np.clip((depth_open_map - 0.68) / 0.18, 0.0, 1.0)

    road_score = (
        0.50
        + 0.18 * narrow_bonus
        + 0.10 * boundary_near
        - 0.22 * vehicle_density
        - 0.05 * light_dynamic_density
        - 0.14 * wide_penalty
        - 0.08 * openness_penalty
    )

    road_score = np.clip(road_score, 0.30, 0.80)
    road_score = road_score * road_mask.astype(np.float32)
    return road_score.astype(np.float32)


def get_binary_mask(label_map: np.ndarray, ids):
    if len(ids) == 0:
        return np.zeros_like(label_map, dtype=np.uint8)
    return np.isin(label_map, ids).astype(np.uint8)


def bottom_connected_component(mask: np.ndarray):
    """
    Keep only regions connected to the bottom row.
    mask: uint8 binary
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    for x in range(w):
        if mask[h - 1, x] > 0:
            visited[h - 1, x] = 1
            q.append((h - 1, x))

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        y, x = q.popleft()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if mask[ny, nx] > 0 and visited[ny, nx] == 0:
                    visited[ny, nx] = 1
                    q.append((ny, nx))

    return visited.astype(np.uint8)


def build_semantic_pack(label_map: np.ndarray, id2label: dict, depth_open_map: np.ndarray):
    groups = find_label_ids(id2label)

    high_walk = get_binary_mask(label_map, groups["high_walk"])
    mid_walk = get_binary_mask(label_map, groups["mid_walk"])
    obstacle = get_binary_mask(label_map, groups["obstacle"])
    vehicle = get_binary_mask(label_map, groups["vehicle"])
    light_dynamic = get_binary_mask(label_map, groups["light_dynamic"])
    sky = get_binary_mask(label_map, groups["sky"])
    building = get_binary_mask(label_map, groups["building"])

    ground_candidate = ((high_walk + mid_walk) > 0).astype(np.uint8)
    ground_candidate = cv2.morphologyEx(
        ground_candidate,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    reachable_ground = bottom_connected_component(ground_candidate)

    # 静态障碍膨胀，不再把 person/bicycle 混进去
    static_obstacle_all = ((obstacle + sky + building) > 0).astype(np.uint8)
    static_obstacle_dilate = cv2.dilate(
        static_obstacle_all,
        np.ones((9, 9), np.uint8),
        iterations=1
    )

    vehicle_dilate = cv2.dilate(
        vehicle,
        np.ones((11, 11), np.uint8),
        iterations=1
    )

    road_refined_score = refine_mid_walk_score(
        mid_walk=mid_walk.astype(np.float32),
        reachable_ground=reachable_ground.astype(np.float32),
        obstacle_mask=obstacle.astype(np.float32),
        building_mask=building.astype(np.float32),
        vehicle_mask=vehicle.astype(np.float32),
        light_dynamic_mask=light_dynamic.astype(np.float32),
        depth_open_map=depth_open_map.astype(np.float32),
    )

    sem_score = np.zeros_like(label_map, dtype=np.float32)
    sem_score[high_walk > 0] = 1.00
    sem_score[mid_walk > 0] = road_refined_score[mid_walk > 0]
    sem_score[light_dynamic > 0] = 0.18
    sem_score[vehicle > 0] = 0.05
    sem_score[obstacle > 0] = 0.00
    sem_score[sky > 0] = 0.00
    sem_score[building > 0] = 0.00

    sem_score = sem_score * reachable_ground.astype(np.float32)

    return {
        "sem_score": sem_score.astype(np.float32),
        "reachable_ground": reachable_ground.astype(np.float32),
        "dynamic_mask": ((vehicle + light_dynamic) > 0).astype(np.float32),
        "vehicle_mask": vehicle.astype(np.float32),
        "light_dynamic_mask": light_dynamic.astype(np.float32),
        "obstacle_mask": obstacle.astype(np.float32),
        "static_obstacle_dilate": static_obstacle_dilate.astype(np.float32),
        "vehicle_dilate": vehicle_dilate.astype(np.float32),
        "high_walk": high_walk.astype(np.float32),
        "mid_walk": mid_walk.astype(np.float32),
        "road_refined_score": road_refined_score.astype(np.float32),
        "sky_mask": sky.astype(np.float32),
        "building_mask": building.astype(np.float32),
    }



def build_raw_heatmap(
    sem_score: np.ndarray,
    depth_open_map: np.ndarray,
    reachable_ground: np.ndarray,
    light_dynamic_mask: np.ndarray,
    vehicle_dilate: np.ndarray,
    static_obstacle_dilate: np.ndarray,
):
    h, w = sem_score.shape

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1).repeat(w, axis=1)

    top_penalty = np.clip((0.58 - yy) / 0.58, 0.0, 1.0) * 0.18

    edge_penalty = np.zeros((h, w), dtype=np.float32)
    edge_w = max(1, int(0.06 * w))
    edge_penalty[:, :edge_w] = 0.08
    edge_penalty[:, -edge_w:] = 0.08

    risk = (
        0.12 * light_dynamic_mask +      # 人/自行车轻惩罚
        0.40 * vehicle_dilate +         # 汽车强惩罚
        0.32 * static_obstacle_dilate + # 静态障碍中等惩罚
        top_penalty +
        edge_penalty
    )

    # depth 权重略降，避免“开阔大马路”天然占优
    raw = 0.80 * sem_score + 0.20 * depth_open_map - risk
    raw = raw * reachable_ground
    raw = np.clip(raw, 0.0, 1.0)
    raw = cv2.GaussianBlur(raw, (7, 7), 0)
    raw = np.clip(raw, 0.0, 1.0)
    return raw.astype(np.float32)


# =========================
# Temporal fusion
# =========================

def to_gray(bgr: np.ndarray):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def compute_backward_flow(curr_bgr: np.ndarray, prev_bgr: np.ndarray):
    """
    Dense optical flow from current frame to previous frame.
    This is convenient for warping prev_map into current frame coordinates.
    """
    curr_gray = to_gray(curr_bgr)
    prev_gray = to_gray(prev_bgr)

    flow = cv2.calcOpticalFlowFarneback(
        curr_gray,
        prev_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def warp_prev_map_to_current(prev_map: np.ndarray, backward_flow: np.ndarray):
    """
    prev_map: H x W float32
    backward_flow: H x W x 2, mapping current -> previous
    """
    h, w = prev_map.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (grid_x + backward_flow[..., 0]).astype(np.float32)
    map_y = (grid_y + backward_flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        prev_map.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid = cv2.remap(
        np.ones((h, w), dtype=np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (valid > 0.5).astype(np.float32)
    return warped.astype(np.float32), valid.astype(np.float32)


def temporal_fuse(
    raw_map: np.ndarray,
    warped_prev_map: np.ndarray,
    valid_mask: np.ndarray,
    dynamic_mask: np.ndarray = None,
    alpha: float = 0.72,
    dynamic_alpha: float = 0.90,
):
    """
    alpha: trust current frame ratio in static area
    dynamic_alpha: trust current frame more in dynamic area
    """
    raw_map = raw_map.astype(np.float32)
    warped_prev_map = warped_prev_map.astype(np.float32)
    valid_mask = valid_mask.astype(np.float32)

    alpha_map = np.full_like(raw_map, alpha, dtype=np.float32)

    if dynamic_mask is not None:
        alpha_map[dynamic_mask > 0.5] = dynamic_alpha

    # invalid warp region should fully trust current frame
    alpha_map[valid_mask < 0.5] = 1.0

    fused = alpha_map * raw_map + (1.0 - alpha_map) * warped_prev_map
    fused = cv2.GaussianBlur(fused, (5, 5), 0)
    fused = np.clip(fused, 0.0, 1.0)
    return fused.astype(np.float32)


def compute_temporal_metrics(raw_map, stable_map, warped_prev_map, valid_mask):
    eps = 1e-6
    valid = valid_mask > 0.5
    if int(valid.sum()) < 10:
        return {
            "raw_delta_to_prev": None,
            "stable_delta_to_prev": None,
            "stability_gain": None,
        }

    raw_delta = float(np.mean(np.abs(raw_map[valid] - warped_prev_map[valid])))
    stable_delta = float(np.mean(np.abs(stable_map[valid] - warped_prev_map[valid])))

    gain = (raw_delta - stable_delta) / (raw_delta + eps)

    return {
        "raw_delta_to_prev": raw_delta,
        "stable_delta_to_prev": stable_delta,
        "stability_gain": float(gain),
    }


# =========================
# Explainability
# =========================

def compute_analysis_metrics(stable_map: np.ndarray, sem_pack: dict):
    high_mask = sem_pack["high_walk"] > 0.5
    mid_mask = sem_pack["mid_walk"] > 0.5
    dyn_mask = sem_pack["dynamic_mask"] > 0.5
    obst_mask = sem_pack["obstacle_mask"] > 0.5
    sky_mask = sem_pack["sky_mask"] > 0.5
    build_mask = sem_pack["building_mask"] > 0.5
    reach_mask = sem_pack["reachable_ground"] > 0.5

    def safe_mean(x, m):
        if m.sum() == 0:
            return None
        return float(x[m].mean())

    return {
        "mean_stable_heat": float(stable_map.mean()),
        "reachable_ratio": float(reach_mask.mean()),
        "high_walk_ratio": float(high_mask.mean()),
        "mid_walk_ratio": float(mid_mask.mean()),
        "dynamic_ratio": float(dyn_mask.mean()),
        "heat_on_high_walk": safe_mean(stable_map, high_mask),
        "heat_on_mid_walk": safe_mean(stable_map, mid_mask),
        "heat_on_dynamic": safe_mean(stable_map, dyn_mask),
        "heat_on_obstacle": safe_mean(stable_map, obst_mask),
        "heat_on_sky": safe_mean(stable_map, sky_mask),
        "heat_on_building": safe_mean(stable_map, build_mask),
    }


def build_explanation_text(sem_pack: dict, temporal_info: dict, stable_metrics: dict):
    parts = []

    high_ratio = stable_metrics["high_walk_ratio"]
    mid_ratio = stable_metrics["mid_walk_ratio"]
    dyn_ratio = stable_metrics["dynamic_ratio"]
    reach_ratio = stable_metrics["reachable_ratio"]

    if high_ratio is not None and high_ratio > 0.08:
        parts.append("heat response is mainly supported by sidewalk/crosswalk-like regions")
    elif mid_ratio is not None and mid_ratio > 0.15:
        parts.append("heat response mainly comes from road-like ground regions")
    else:
        parts.append("the current frame contains limited confident walkable area")

    if dyn_ratio is not None and dyn_ratio > 0.03:
        parts.append("dynamic objects are down-weighted to suppress unstable high response")

    if reach_ratio is not None and reach_ratio < 0.25:
        parts.append("only bottom-connected ground is retained, which strongly suppresses sky and building regions")

    heat_on_sky = stable_metrics["heat_on_sky"]
    heat_on_building = stable_metrics["heat_on_building"]

    if heat_on_sky is not None and heat_on_sky < 0.05:
        parts.append("sky regions are effectively suppressed")
    if heat_on_building is not None and heat_on_building < 0.08:
        parts.append("building regions are effectively suppressed")

    gain = temporal_info.get("stability_gain", None)
    if gain is not None:
        if gain > 0.15:
            parts.append("temporal fusion significantly reduces frame-to-frame jitter")
        elif gain > 0.05:
            parts.append("temporal fusion slightly improves temporal consistency")
        else:
            parts.append("temporal fusion has limited effect in this frame")

    return "; ".join(parts) + "."


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seg_model_dir", type=str, required=True)
    parser.add_argument("--depth_model_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--long_side", type=int, default=1280)

    parser.add_argument(
        "--depth_mode",
        type=str,
        default="near_is_high",
        choices=["near_is_high", "far_is_high"],
        help="Use near_is_high if your saved grayscale depth usually shows near objects brighter."
    )

    parser.add_argument("--temporal_alpha", type=float, default=0.72)
    parser.add_argument("--dynamic_alpha", type=float, default=0.90)
    parser.add_argument("--save_npy", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "semseg"))
    ensure_dir(os.path.join(args.output_dir, "depth"))
    ensure_dir(os.path.join(args.output_dir, "heatmap_raw"))
    ensure_dir(os.path.join(args.output_dir, "heatmap_stable"))
    ensure_dir(os.path.join(args.output_dir, "overlay"))
    ensure_dir(os.path.join(args.output_dir, "explain"))

    if args.save_npy:
        ensure_dir(os.path.join(args.output_dir, "raw_npy"))
        ensure_dir(os.path.join(args.output_dir, "stable_npy"))

    image_paths = sorted_image_list(args.input_dir)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    print("[INFO] Loading models...")
    seg_processor, seg_model = load_segmentation_model(args.seg_model_dir, args.device)
    depth_processor, depth_model = load_depth_model(args.depth_model_dir, args.device)

    prev_bgr = None
    prev_stable_map = None

    print(f"[INFO] Found {len(image_paths)} frames.")
    print(f"[INFO] Device: {args.device}")

    for idx, image_path in enumerate(tqdm(image_paths, desc="Processing")):
        name = os.path.splitext(os.path.basename(image_path))[0]

        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"[WARN] Failed to read: {image_path}")
            continue

        bgr = resize_keep_aspect(bgr, args.long_side)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1) semantic segmentation
        label_map, id2label, color_seg = predict_semantic(
            rgb=rgb,
            processor=seg_processor,
            model=seg_model,
            device=args.device,
        )

        # 2) depth estimation
        depth_open_map, depth_vis, depth_norm_source = predict_depth(
            rgb=rgb,
            processor=depth_processor,
            model=depth_model,
            device=args.device,
            depth_mode=args.depth_mode,
        )

        # 3) semantic pack + raw walkability heatmap
        sem_pack = build_semantic_pack(label_map, id2label, depth_open_map)

        raw_map = build_raw_heatmap(
            sem_score=sem_pack["sem_score"],
            depth_open_map=depth_open_map,
            reachable_ground=sem_pack["reachable_ground"],
            light_dynamic_mask=sem_pack["light_dynamic_mask"],
            vehicle_dilate=sem_pack["vehicle_dilate"],
            static_obstacle_dilate=sem_pack["static_obstacle_dilate"],
        )


        # 4) temporal fusion
        if prev_bgr is None or prev_stable_map is None:
            stable_map = raw_map.copy()
            temporal_info = {
                "raw_delta_to_prev": None,
                "stable_delta_to_prev": None,
                "stability_gain": None,
            }
        else:
            backward_flow = compute_backward_flow(
                curr_bgr=bgr,
                prev_bgr=prev_bgr,
            )
            warped_prev_map, valid_mask = warp_prev_map_to_current(
                prev_map=prev_stable_map,
                backward_flow=backward_flow,
            )

            stable_map = temporal_fuse(
                raw_map=raw_map,
                warped_prev_map=warped_prev_map,
                valid_mask=valid_mask,
                dynamic_mask=sem_pack["dynamic_mask"],
                alpha=args.temporal_alpha,
                dynamic_alpha=args.dynamic_alpha,
            )

            temporal_info = compute_temporal_metrics(
                raw_map=raw_map,
                stable_map=stable_map,
                warped_prev_map=warped_prev_map,
                valid_mask=valid_mask,
            )

        # 5) save visualizations
        cv2.imwrite(os.path.join(args.output_dir, "semseg", f"{name}.png"), color_seg)
        cv2.imwrite(os.path.join(args.output_dir, "depth", f"{name}.png"), depth_vis)

        save_heatmap_png(
            raw_map,
            os.path.join(args.output_dir, "heatmap_raw", f"{name}.png")
        )
        save_heatmap_png(
            stable_map,
            os.path.join(args.output_dir, "heatmap_stable", f"{name}.png")
        )

        overlay = overlay_heatmap_on_bgr(bgr, stable_map, alpha=0.42)
        cv2.imwrite(os.path.join(args.output_dir, "overlay", f"{name}.png"), overlay)

        if args.save_npy:
            np.save(os.path.join(args.output_dir, "raw_npy", f"{name}.npy"), raw_map.astype(np.float32))
            np.save(os.path.join(args.output_dir, "stable_npy", f"{name}.npy"), stable_map.astype(np.float32))

        # 6) explainability json
        stable_metrics = compute_analysis_metrics(stable_map, sem_pack)
        explain = {
            "frame_name": name,
            "temporal": temporal_info,
            "analysis": stable_metrics,
            "explanation_text": build_explanation_text(
                sem_pack=sem_pack,
                temporal_info=temporal_info,
                stable_metrics=stable_metrics,
            ),
        }

        with open(
            os.path.join(args.output_dir, "explain", f"{name}.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(explain, f, ensure_ascii=False, indent=2)

        # update state
        prev_bgr = bgr.copy()
        prev_stable_map = stable_map.copy()

    print("[INFO] Done.")
    print(f"[INFO] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


    """
    python run_demo.py --input_dir ../Data/Videos_Frame/7 --output_dir ./outputs --seg_model_dir ./models/mask2former-mapillary --depth_model_dir ./models/depth-anything-v2-base
    """