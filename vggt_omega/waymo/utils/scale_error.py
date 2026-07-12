from __future__ import annotations

import numpy as np

from vggt_omega.waymo.types import WaymoBox3D, WaymoFrame
from vggt_omega.waymo.geometry import PinholeCamera
from vggt_omega.waymo.utils.lidar_prompts import lidar_points_in_box_ego

DEFAULT_SCALE_ERROR_THRESHOLD = 0.08
DEFAULT_MIN_INBOX_POINTS = 5
DEFAULT_MIN_DEPTH_M = 1.0


def sample_pred_depth_at_uv(
    pred_depth: np.ndarray,
    uv: np.ndarray,
    camera: PinholeCamera,
) -> np.ndarray:
    depth_map = pred_depth[..., 0] if pred_depth.ndim == 3 else pred_depth
    pred_h, pred_w = depth_map.shape
    u = np.clip(np.round(uv[:, 0] * pred_w / camera.width_px).astype(np.int32), 0, pred_w - 1)
    v = np.clip(np.round(uv[:, 1] * pred_h / camera.height_px).astype(np.int32), 0, pred_h - 1)
    return depth_map[v, u]


def box_depth_scale_error(
    box: WaymoBox3D,
    frame: WaymoFrame,
    camera: PinholeCamera,
    pred_depth: np.ndarray,
    *,
    min_points: int = DEFAULT_MIN_INBOX_POINTS,
    min_depth: float = DEFAULT_MIN_DEPTH_M,
) -> float | None:
    lidar_ego = lidar_points_in_box_ego(frame, box, expand_ratio=1.0)
    if len(lidar_ego) < min_points:
        return None

    cam_se3_ego = camera.ego_SE3_cam.inverse()
    points_cam = cam_se3_ego.transform_point_cloud(lidar_ego)
    uv, points_cam_proj, valid = camera.project_cam_to_img(points_cam)
    keep = (
        valid
        & (points_cam_proj[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < camera.width_px)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera.height_px)
    )
    if keep.sum() < min_points:
        return None

    z_lidar = points_cam_proj[keep, 2]
    z_pred = sample_pred_depth_at_uv(pred_depth, uv[keep], camera)
    valid_depth = np.isfinite(z_pred) & (z_pred > min_depth) & (z_lidar > min_depth)
    if valid_depth.sum() < min_points:
        return None

    return float(np.median(np.abs(z_lidar[valid_depth] - z_pred[valid_depth]) / z_lidar[valid_depth]))
