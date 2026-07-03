from __future__ import annotations

from pathlib import Path

import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.camera.pinhole_camera import Intrinsics, PinholeCamera

from vggt_omega.av2.dataset import AV2Frame, AV2SceneDataset, FRONT_CAMERA


def build_pinhole_camera(
    data_root: str | Path,
    frame: AV2Frame,
    crop_bottom: int = 0,
) -> PinholeCamera:
    """Load AV2 camera calibration and optionally adjust it for bottom cropping."""
    camera = PinholeCamera.from_feather(Path(data_root) / frame.log_id, FRONT_CAMERA)
    if crop_bottom <= 0:
        return camera

    intrinsics = camera.intrinsics
    cropped_height = intrinsics.height_px - crop_bottom
    if cropped_height <= 0:
        raise ValueError(f"crop_bottom={crop_bottom} removes the full image")

    return PinholeCamera(
        ego_SE3_cam=camera.ego_SE3_cam,
        intrinsics=Intrinsics(
            fx_px=intrinsics.fx_px,
            fy_px=intrinsics.fy_px,
            cx_px=intrinsics.cx_px,
            cy_px=intrinsics.cy_px,
            width_px=intrinsics.width_px,
            height_px=cropped_height,
        ),
        cam_name=camera.cam_name,
    )


def scale_pinhole_camera(
    camera: PinholeCamera,
    width_px: int,
    height_px: int,
) -> PinholeCamera:
    """Scale camera intrinsics to match a resized image grid."""
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
    """Back-project metric depth to camera-frame points using AV2 intrinsics."""
    depth_map = depth[..., 0] if depth.ndim == 3 else depth
    height, width = depth_map.shape
    intrinsics = camera.intrinsics.K

    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    u_av2 = u.astype(float).ravel() * camera.width_px / width
    v_av2 = v.astype(float).ravel() * camera.height_px / height
    z = depth_map.ravel()

    x = (u_av2 - intrinsics[0, 2]) / intrinsics[0, 0] * z
    y = (v_av2 - intrinsics[1, 2]) / intrinsics[1, 1] * z
    points_cam = np.stack([x, y, z], axis=1)
    uv = np.stack([u_av2, v_av2], axis=1)

    valid = np.isfinite(points_cam).all(axis=1) & (z > 0)
    return points_cam[valid], uv[valid]


def motion_compensate_ego(
    points_ego: np.ndarray,
    city_SE3_ego_src,
    city_SE3_ego_dst,
) -> np.ndarray:
    """Move ego-frame points from one capture time to another."""
    ego_dst_SE3_ego_src = city_SE3_ego_dst.inverse().compose(city_SE3_ego_src)
    return ego_dst_SE3_ego_src.transform_point_cloud(points_ego)


def motion_compensate_lidar_to_camera_time(
    frame: AV2Frame,
    lidar_xyz_ego: np.ndarray,
    loader: AV2SensorDataLoader,
) -> np.ndarray:
    """Move ego-frame LiDAR points from sweep time to camera capture time."""
    city_SE3_ego_lidar = loader.get_city_SE3_ego(frame.log_id, frame.lidar_timestamp_ns)
    return motion_compensate_ego(lidar_xyz_ego, city_SE3_ego_lidar, frame.city_SE3_ego)


def project_lidar_to_depth_map(
    camera: PinholeCamera,
    lidar_ego: np.ndarray,
) -> np.ndarray:
    """Rasterize LiDAR into a camera-frame depth map (closest z per pixel)."""
    uv, keep = project_ego_points_to_uv(camera, lidar_ego)
    if uv.size == 0:
        return np.full((camera.height_px, camera.width_px), np.nan, dtype=np.float32)

    cam_SE3_ego = camera.ego_SE3_cam.inverse()
    points_cam = cam_SE3_ego.transform_point_cloud(lidar_ego)
    z = points_cam[keep, 2]
    uv_int = np.round(uv).astype(np.int32)

    lin_idx = uv_int[:, 1] * camera.width_px + uv_int[:, 0]
    order = np.argsort(z)
    lin_idx = lin_idx[order]
    z = z[order]
    _, first = np.unique(lin_idx, return_index=True)

    depth = np.full((camera.height_px, camera.width_px), np.nan, dtype=np.float32)
    depth.ravel()[lin_idx[first]] = z[first]
    return depth


def project_ego_points_to_uv(
    camera: PinholeCamera,
    points_ego: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame 3D points to image pixels visible in the camera."""
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


def sample_depth_map_to_prediction_grid(
    depth_map: np.ndarray,
    camera: PinholeCamera,
    pred_height: int,
    pred_width: int,
) -> np.ndarray:
    """Resample an AV2-resolution depth map onto the VGGT prediction grid."""
    v, u = np.meshgrid(np.arange(pred_height), np.arange(pred_width), indexing="ij")
    u_av2 = np.clip(
        np.round(u.astype(float) * camera.width_px / pred_width).astype(np.int32),
        0,
        camera.width_px - 1,
    )
    v_av2 = np.clip(
        np.round(v.astype(float) * camera.height_px / pred_height).astype(np.int32),
        0,
        camera.height_px - 1,
    )
    return depth_map[v_av2, u_av2]


def estimate_scale_from_depth_maps(
    pred_depth: np.ndarray,
    lidar_depth: np.ndarray,
    *,
    percentile: float = 90.0,
    min_depth: float = 1.0,
) -> float:
    """Estimate metric scale as the median lidar/pred depth ratio on overlapping pixels."""
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
    frame: AV2Frame,
    data_root: str | Path,
    *,
    crop_bottom: int = 0,
    percentile: float = 90.0,
) -> float:
    """Estimate metric scale by projecting LiDAR onto the image and comparing depth maps."""
    data_root = Path(data_root)
    camera = build_pinhole_camera(data_root, frame, crop_bottom)
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    pred_depth = predictions["depth"][0]
    pred_height, pred_width = pred_depth.shape[:2]
    camera = scale_pinhole_camera(camera, pred_width, pred_height)

    sweep = AV2SceneDataset.load_sweep(frame.lidar_path)
    lidar_ego = motion_compensate_lidar_to_camera_time(frame, sweep.xyz, loader)
    lidar_depth = project_lidar_to_depth_map(camera, lidar_ego)

    return estimate_scale_from_depth_maps(
        pred_depth,
        lidar_depth,
        percentile=percentile,
    )


def apply_metric_scale(predictions: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    """Scale all predicted depths by the estimated metric factor."""
    aligned = dict(predictions)
    aligned["depth"] = aligned["depth"] * scale
    aligned["metric_scale"] = np.array(scale)
    return aligned
