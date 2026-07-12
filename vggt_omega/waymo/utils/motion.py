from __future__ import annotations

from collections import defaultdict

import numpy as np

from vggt_omega.waymo.types import WaymoBox3D, WaymoFrame
from vggt_omega.waymo.utils.dynamic_boxes import dynamic_boxes

DEFAULT_MIN_BOX_DISPLACEMENT_M = 0.2


def box_center_world(frame: WaymoFrame, box: WaymoBox3D) -> np.ndarray:
    center_ego = box.ego_SE3_object[:3, 3]
    return frame.world_SE3_ego.transform_point_cloud(center_ego[None, :])[0]


def moving_track_ids(
    frames: list[WaymoFrame],
    *,
    min_displacement_m: float = DEFAULT_MIN_BOX_DISPLACEMENT_M,
) -> frozenset[str]:
    centers_by_track: dict[str, list[np.ndarray]] = defaultdict(list)
    for frame in frames:
        for box in dynamic_boxes(frame):
            centers_by_track[box.track_id].append(box_center_world(frame, box))

    moving: set[str] = set()
    for track_id, centers in centers_by_track.items():
        if len(centers) < 2:
            continue
        path = np.stack(centers)
        if float(np.linalg.norm(path[-1] - path[0])) > min_displacement_m:
            moving.add(track_id)
    return frozenset(moving)
