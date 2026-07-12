from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vggt_omega.waymo.types import WaymoBox3D, WaymoFrame
from vggt_omega.waymo.geometry import PinholeCamera
from vggt_omega.waymo.metric_alignment import project_ego_points_to_uv
from vggt_omega.waymo.utils.dynamic_boxes import (
    points_inside_box,
    project_box_to_xyxy,
)

DEFAULT_MAX_LIDAR_PROMPT_POINTS = 24
MAX_LIDAR_POINTS_FOR_BOX_QUERY = 80_000


@dataclass(frozen=True)
class DynamicObjectSamPrompt:
    box: WaymoBox3D
    xyxy: np.ndarray
    category: str
    lidar_uv: np.ndarray
    scale_error: float | None = None
    use_3d_box: bool = False


def load_frame_image_rgb(image_path: Path | str, *, crop_bottom: int = 0) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_bottom > 0:
        image = image[: image.shape[0] - crop_bottom]
    return image


def subsample_uv_points(uv: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if len(uv) <= max_points:
        return uv
    rng = np.random.default_rng(seed)
    return uv[rng.choice(len(uv), size=max_points, replace=False)]


def lidar_points_in_box_ego(
    frame: WaymoFrame,
    box: WaymoBox3D,
    *,
    expand_ratio: float = 1.0,
) -> np.ndarray:
    xyz = frame.lidar_xyz_ego
    if len(xyz) > MAX_LIDAR_POINTS_FOR_BOX_QUERY:
        rng = np.random.default_rng(hash(box.track_id) % (2**32))
        xyz = xyz[rng.choice(len(xyz), size=MAX_LIDAR_POINTS_FOR_BOX_QUERY, replace=False)]
    return xyz[points_inside_box(xyz, box, expand_ratio=expand_ratio)]


def sam_prompt_for_box(
    frame: WaymoFrame,
    box: WaymoBox3D,
    xyxy: np.ndarray,
    camera: PinholeCamera,
    *,
    max_lidar_points: int = DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    scale_error: float | None = None,
    use_3d_box: bool = False,
) -> DynamicObjectSamPrompt:
    lidar_ego = lidar_points_in_box_ego(frame, box, expand_ratio=1.0)
    lidar_uv, _ = project_ego_points_to_uv(camera, lidar_ego)
    tight_xyxy = project_box_to_xyxy(box, camera, expand_ratio=1.0)
    filter_xyxy = tight_xyxy if tight_xyxy is not None else xyxy
    x0, y0, x1, y1 = filter_xyxy
    in_2d_box = (
        (lidar_uv[:, 0] >= x0)
        & (lidar_uv[:, 0] <= x1)
        & (lidar_uv[:, 1] >= y0)
        & (lidar_uv[:, 1] <= y1)
    )
    lidar_uv = lidar_uv[in_2d_box]
    lidar_uv = subsample_uv_points(lidar_uv, max_lidar_points, seed=hash(box.track_id) % (2**32))
    return DynamicObjectSamPrompt(
        box=box,
        xyxy=xyxy,
        category=box.category,
        lidar_uv=lidar_uv,
        scale_error=scale_error,
        use_3d_box=use_3d_box,
    )
