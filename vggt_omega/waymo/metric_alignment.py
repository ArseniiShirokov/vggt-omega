from __future__ import annotations

from pathlib import Path

import numpy as np

from vggt_omega.waymo.types import WaymoFrame
from vggt_omega.waymo.geometry import Intrinsics, PinholeCamera, SE3


def build_pinhole_camera(
    frame: WaymoFrame,
    crop_bottom: int = 0,
) -> PinholeCamera:
    """Build a pinhole camera for a Waymo frame, optionally cropping the bottom."""
    intrinsics = frame.intrinsics
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    width_px = frame.image_width
    height_px = frame.image_height - crop_bottom
    if height_px <= 0:
        raise ValueError(f"crop_bottom={crop_bottom} removes the full image")

    return PinholeCamera(
        ego_SE3_cam=SE3.from_matrix(frame.ego_SE3_cam),
        intrinsics=Intrinsics(
            fx_px=float(fx),
            fy_px=float(fy),
            cx_px=float(cx),
            cy_px=float(cy),
            width_px=int(width_px),
            height_px=int(height_px),
        ),
        cam_name="FRONT",
    )


def scale_pinhole_camera(
    camera: PinholeCamera,
    width_px: int,
    height_px: int,
) -> PinholeCamera:
    intrinsics = camera.intrinsics
    scale_x = width_px / intrinsics.width_px
    scale_y = height_px / intrinsics.height_px
    return PinholeCamera(
        ego_SE3_cam=camera.ego_SE3_cam,
        intrinsics=Intrinsics(
            fx_px=intrinsics.fx_px * scale_x,
            fy_px=intrinsics.fy_px * scale_y,
            cx_px=intrinsics.cx_px * scale_x,
            cy_px=intrinsics.cy_px * scale_y,
            width_px=width_px,
            height_px=height_px,
        ),
        cam_name=camera.cam_name,
    )


def depth_to_cam_points(
    depth: np.ndarray,
    camera: PinholeCamera,
) -> tuple[np.ndarray, np.ndarray]:
    depth_map = depth[..., 0] if depth.ndim == 3 else depth
    height, width = depth_map.shape
    k = camera.intrinsics.K

    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    u_native = u.astype(float).ravel() * camera.width_px / width
    v_native = v.astype(float).ravel() * camera.height_px / height
    z = depth_map.ravel()

    x = (u_native - k[0, 2]) / k[0, 0] * z
    y = (v_native - k[1, 2]) / k[1, 1] * z
    points_cam = np.stack([x, y, z], axis=1)
    uv = np.stack([u_native, v_native], axis=1)

    valid = np.isfinite(points_cam).all(axis=1) & (z > 0)
    return points_cam[valid], uv[valid]


def motion_compensate_ego(
    points_ego: np.ndarray,
    world_SE3_ego_src: SE3,
    world_SE3_ego_dst: SE3,
) -> np.ndarray:
    ego_dst_SE3_ego_src = world_SE3_ego_dst.inverse().compose(world_SE3_ego_src)
    return ego_dst_SE3_ego_src.transform_point_cloud(points_ego)


def project_ego_points_to_uv(
    camera: PinholeCamera,
    points_ego: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points_ego) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    cam_SE3_ego = camera.ego_SE3_cam.inverse()
    points_cam = cam_SE3_ego.transform_point_cloud(points_ego)
    uv, points_cam, valid = camera.project_cam_to_img(points_cam)
    keep = (
        valid
        & (points_cam[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < camera.width_px)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera.height_px)
    )
    return uv[keep].astype(np.float32), keep


def project_lidar_to_depth_map(
    camera: PinholeCamera,
    lidar_ego: np.ndarray,
) -> np.ndarray:
    uv, keep = project_ego_points_to_uv(camera, lidar_ego)
    if uv.size == 0:
        return np.full((camera.height_px, camera.width_px), np.nan, dtype=np.float32)

    cam_SE3_ego = camera.ego_SE3_cam.inverse()
    points_cam = cam_SE3_ego.transform_point_cloud(lidar_ego)
    z = points_cam[keep, 2]
    uv_int = np.round(uv).astype(np.int32)
    u, v = uv_int[:, 0], uv_int[:, 1]

    in_bounds = (u >= 0) & (u < camera.width_px) & (v >= 0) & (v < camera.height_px) & (z > 0)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]
    if len(z) == 0:
        return np.full((camera.height_px, camera.width_px), np.nan, dtype=np.float32)

    lin_idx = v.astype(np.int64) * camera.width_px + u.astype(np.int64)
    order = np.lexsort((z, lin_idx))
    lin_sorted = lin_idx[order]
    _, nearest = np.unique(lin_sorted, return_index=True)

    depth = np.full((camera.height_px, camera.width_px), np.nan, dtype=np.float32)
    depth[v[order][nearest], u[order][nearest]] = z[order][nearest]
    return depth


def estimate_scale_from_depth_maps(
    pred_depth: np.ndarray,
    lidar_depth: np.ndarray,
    *,
    percentile: float = 90.0,
    min_depth: float = 1.0,
) -> float:
    pred_z = pred_depth[..., 0] if pred_depth.ndim == 3 else pred_depth
    valid = (
        np.isfinite(pred_z)
        & (pred_z > min_depth)
        & np.isfinite(lidar_depth)
        & (lidar_depth > min_depth)
    )
    ratios = lidar_depth[valid] / pred_z[valid]
    if ratios.size == 0:
        raise RuntimeError("No overlapping pred/LiDAR depth pixels for scale estimation")

    cutoff = np.percentile(ratios, percentile)
    ratios = ratios[ratios <= cutoff]
    return float(np.median(ratios))


def compute_metric_scale_from_first_frame(
    predictions: dict[str, np.ndarray],
    frame: WaymoFrame,
    *,
    crop_bottom: int = 0,
    percentile: float = 90.0,
) -> float:
    camera = build_pinhole_camera(frame, crop_bottom)
    pred_depth = predictions["depth"][0]
    pred_height, pred_width = pred_depth.shape[:2]
    camera = scale_pinhole_camera(camera, pred_width, pred_height)

    lidar_depth = project_lidar_to_depth_map(camera, frame.lidar_xyz_ego)
    return estimate_scale_from_depth_maps(pred_depth, lidar_depth, percentile=percentile)


def apply_metric_scale(predictions: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    aligned = dict(predictions)
    aligned["depth"] = aligned["depth"] * scale
    aligned["metric_scale"] = np.array(scale)
    return aligned
