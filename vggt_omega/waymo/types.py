from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vggt_omega.waymo.geometry import SE3

WAYMO_TYPE_NAMES: dict[int, str] = {
    0: "UNKNOWN",
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "SIGN",
    4: "CYCLIST",
}

DYNAMIC_TYPE_IDS = frozenset({1, 2, 4})


@dataclass(frozen=True)
class WaymoBox3D:
    track_id: str
    type_id: int
    category: str
    length_m: float
    width_m: float
    height_m: float
    ego_SE3_object: np.ndarray


@dataclass(frozen=True)
class WaymoFrame:
    """Single front-camera frame with aligned LiDAR in ego coordinates."""

    scene_id: str
    sample_idx: int
    timestamp_us: int
    image_path: Path
    intrinsics: np.ndarray
    ego_SE3_cam: np.ndarray
    world_SE3_ego: SE3
    lidar_xyz_ego: np.ndarray
    image_width: int
    image_height: int
    boxes: tuple[WaymoBox3D, ...] = ()
