from __future__ import annotations

import numpy as np
import torch

from vggt_omega.waymo.geometry import SE3

TOP_LIDAR_IDX = 1
SIDE_LIDAR_IDXS = [2, 3, 4, 5]
ALL_LIDAR_IDXS = [TOP_LIDAR_IDX] + SIDE_LIDAR_IDXS
TOP_LIDAR_SWEEP = 2650
TOP_LIDAR_RAYS = 64
SIDE_LIDAR_SWEEP = 600
SIDE_LIDAR_RAYS = 200


def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert roll-pitch-yaw (XYZ) angles to rotation matrices."""
    x, y, z = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    rot = torch.empty((*euler.shape[:-1], 3, 3), dtype=euler.dtype, device=euler.device)
    rot[..., 0, 0] = cy * cz
    rot[..., 0, 1] = -cy * sz
    rot[..., 0, 2] = sy
    rot[..., 1, 0] = sx * sy * cz + cx * sz
    rot[..., 1, 1] = -sx * sy * sz + cx * cz
    rot[..., 1, 2] = -sx * cy
    rot[..., 2, 0] = -cx * sy * cz + sx * sz
    rot[..., 2, 1] = cx * sy * sz + sx * cz
    rot[..., 2, 2] = cx * cy
    return rot


def _apply_pose_batch(rotation: torch.Tensor, translation: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", rotation, points) + translation


def _rng_image(row: dict[str, np.ndarray], component: str = "LiDARComponent", return_idx: int = 1) -> torch.Tensor:
    key_template = "[{}].range_image_return{}.{}"
    shape = row[key_template.format(component, return_idx, "shape")].tolist()
    values = row[key_template.format(component, return_idx, "values")].reshape(*shape)
    return torch.as_tensor(values.copy())


def _get_inclinations(calib: dict[str, object]) -> torch.Tensor:
    beam_incls = calib["beam_incls"]
    if beam_incls is not None:
        return beam_incls
    height = int(calib["height"])
    incl_min = float(calib["beam_incl_min"])
    incl_max = float(calib["beam_incl_max"])
    return torch.arange(height).add(0.5).mul((incl_max - incl_min) / height).add(incl_min)


def precompute_lidar_rays(calib: dict[str, object]) -> torch.Tensor:
    height, width = int(calib["height"]), int(calib["width"])
    inclination = torch.flip(_get_inclinations(calib), dims=[0]).float()
    extrinsic = calib["extrinsic"]
    az_correction = torch.atan2(extrinsic[1, 0], extrinsic[0, 0])
    azimuth = torch.arange(width, 0, -1) * (2 * torch.pi / width) - ((width + 1) * torch.pi / width + az_correction)
    azimuth_tile = azimuth.view(1, -1).expand(height, -1)
    incl_tile = inclination.view(-1, 1).expand(-1, width)
    cos_azimuth, sin_azimuth = azimuth_tile.cos(), azimuth_tile.sin()
    cos_incl, sin_incl = incl_tile.cos(), incl_tile.sin()
    return torch.stack([cos_azimuth * cos_incl, sin_azimuth * cos_incl, sin_incl], dim=-1)


def process_lidar_calib(calib_df) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    extr_key = "[LiDARCalibrationComponent].extrinsic.transform"
    for _, row in calib_df.iterrows():
        laser_name = int(row["key.laser_name"])
        extrinsic = torch.as_tensor(row[extr_key].copy()).view(4, 4)
        beam_incls = row["[LiDARCalibrationComponent].beam_inclination.values"]
        beam_incls = torch.as_tensor(beam_incls.copy()) if beam_incls is not None else None
        calib = {
            "laser_name": laser_name,
            "extrinsic": extrinsic,
            "beam_incl_min": row["[LiDARCalibrationComponent].beam_inclination.min"],
            "beam_incl_max": row["[LiDARCalibrationComponent].beam_inclination.max"],
            "beam_incls": beam_incls,
            "width": TOP_LIDAR_SWEEP if laser_name == TOP_LIDAR_IDX else SIDE_LIDAR_SWEEP,
            "height": TOP_LIDAR_RAYS if laser_name == TOP_LIDAR_IDX else SIDE_LIDAR_RAYS,
            "ego_se3_lidar": SE3.from_matrix(extrinsic.float().numpy()),
        }
        calib["lidar_rays"] = precompute_lidar_rays(calib)
        result[laser_name] = calib
    return result


def top_lidar_points_world(
    lidar_calib: dict[str, object],
    rng_image: torch.Tensor,
    pose_image: torch.Tensor,
) -> np.ndarray:
    points_lidar = lidar_calib["lidar_rays"] * rng_image[..., 0:1]
    mask = rng_image[..., 0] > 0
    ego_se3_lidar = lidar_calib["ego_se3_lidar"]
    points_vehicle = ego_se3_lidar.transform_point_cloud(points_lidar[mask].numpy())

    pose_euler = pose_image[..., :3][mask]
    pose_translation = pose_image[..., 3:][mask]
    rotation = _euler_xyz_to_matrix(pose_euler)
    points_world = _apply_pose_batch(
        rotation,
        pose_translation,
        torch.as_tensor(points_vehicle, dtype=pose_euler.dtype),
    )
    return points_world.numpy()


def side_lidar_points_world(
    lidar_calib: dict[str, object],
    rng_image: torch.Tensor,
    world_se3_ego: SE3,
) -> np.ndarray:
    points_lidar = lidar_calib["lidar_rays"] * rng_image[..., 0:1]
    mask = rng_image[..., 0] > 0
    world_se3_lidar = world_se3_ego.compose(lidar_calib["ego_se3_lidar"])
    return world_se3_lidar.transform_point_cloud(points_lidar[mask].numpy())


def load_lidar_xyz_ego(
    lidar_calib: dict[int, dict[str, object]],
    lidar_row: dict[str, np.ndarray],
    pose_row: dict[str, np.ndarray] | None,
    world_se3_ego: SE3,
    *,
    use_side_lidar: bool,
    side_rows: list[dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    world_se3_ego_inv = world_se3_ego.inverse()
    rng = _rng_image(lidar_row, "LiDARComponent")
    pose_img = _rng_image(pose_row, "LiDARPoseComponent") if pose_row is not None else None
    if pose_img is None:
        raise ValueError("Top LiDAR requires pose row")
    points = [
        world_se3_ego_inv.transform_point_cloud(
            top_lidar_points_world(lidar_calib[TOP_LIDAR_IDX], rng, pose_img)
        )
    ]
    if use_side_lidar and side_rows:
        for lidar_idx, side_row in side_rows:
            side_rng = _rng_image(side_row, "LiDARComponent")
            points.append(
                world_se3_ego_inv.transform_point_cloud(
                    side_lidar_points_world(lidar_calib[lidar_idx], side_rng, world_se3_ego)
                )
            )
    return np.concatenate(points, axis=0)
