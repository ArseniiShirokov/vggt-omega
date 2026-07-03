from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.utils.io import read_img

from vggt_omega.av2.dataset import AV2Box3D, AV2Frame, AV2SceneDataset
from vggt_omega.av2.metric_alignment import motion_compensate_lidar_to_camera_time, project_ego_points_to_uv
from vggt_omega.av2.utils.dynamic_boxes import points_inside_box, project_dynamic_boxes_labeled

DEFAULT_MAX_LIDAR_PROMPT_POINTS = 32


@dataclass(frozen=True)
class DynamicObjectSamPrompt:
    box: AV2Box3D
    xyxy: np.ndarray
    category: str
    lidar_uv: np.ndarray
    scale_error: float | None = None
    use_3d_box: bool = False


def load_frame_image_rgb(image_path: Path | str, *, crop_bottom: int = 0) -> np.ndarray:
    image = read_img(Path(image_path), channel_order="RGB")
    if crop_bottom > 0:
        image = image[: image.shape[0] - crop_bottom]
    return image


def subsample_uv_points(uv: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if len(uv) <= max_points:
        return uv
    rng = np.random.default_rng(seed)
    return uv[rng.choice(len(uv), size=max_points, replace=False)]


def lidar_points_in_box_ego(
    frame: AV2Frame,
    box: AV2Box3D,
    loader: AV2SensorDataLoader,
    *,
    expand_ratio: float = 1.0,
) -> np.ndarray:
    sweep = AV2SceneDataset.load_sweep(frame.lidar_path)
    lidar_ego = motion_compensate_lidar_to_camera_time(frame, sweep.xyz, loader)
    return lidar_ego[points_inside_box(lidar_ego, box, expand_ratio=expand_ratio)]


def sam_prompt_for_box(
    frame: AV2Frame,
    box: AV2Box3D,
    xyxy: np.ndarray,
    camera: PinholeCamera,
    loader: AV2SensorDataLoader,
    *,
    max_lidar_points: int = DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    scale_error: float | None = None,
    use_3d_box: bool = False,
) -> DynamicObjectSamPrompt:
    lidar_ego = lidar_points_in_box_ego(frame, box, loader, expand_ratio=1.0)
    lidar_uv, _ = project_ego_points_to_uv(camera, lidar_ego)
    lidar_uv = subsample_uv_points(lidar_uv, max_lidar_points, seed=hash(box.track_uuid) % (2**32))
    return DynamicObjectSamPrompt(
        box=box,
        xyxy=xyxy,
        category=box.category,
        lidar_uv=lidar_uv,
        scale_error=scale_error,
        use_3d_box=use_3d_box,
    )
