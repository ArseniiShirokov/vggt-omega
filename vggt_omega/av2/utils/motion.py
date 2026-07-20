from __future__ import annotations

from collections import defaultdict

import numpy as np

from vggt_omega.av2.dataset import AV2Box3D, AV2Frame
from vggt_omega.av2.utils.dynamic_boxes import dynamic_boxes

DEFAULT_MIN_BOX_DISPLACEMENT_M = 0.2


def box_center_city(frame: AV2Frame, box: AV2Box3D) -> np.ndarray:
    """Box center in city coordinates at this frame's camera time."""
    center_ego = box.ego_SE3_object[:3, 3]
    return frame.city_SE3_ego.transform_point_cloud(center_ego[None, :])[0]


def moving_track_uuids(
    frames: list[AV2Frame],
    *,
    min_displacement_m: float = DEFAULT_MIN_BOX_DISPLACEMENT_M,
) -> frozenset[str]:
    """Tracks whose box center moves more than min_displacement_m in city frame over the chunk.

    A non-positive threshold disables the movement check: every track with a dynamic-category
    box is treated as moving (including tracks seen in only one frame of the chunk), so all
    dynamic-class objects get filtered.
    """
    centers_by_track: dict[str, list[np.ndarray]] = defaultdict(list)
    for frame in frames:
        for box in dynamic_boxes(frame):
            centers_by_track[box.track_uuid].append(box_center_city(frame, box))

    if min_displacement_m <= 0.0:
        return frozenset(centers_by_track.keys())

    moving: set[str] = set()
    for track_uuid, centers in centers_by_track.items():
        if len(centers) < 2:
            continue
        path = np.stack(centers)
        if float(np.linalg.norm(path[-1] - path[0])) > min_displacement_m:
            moving.add(track_uuid)
    return frozenset(moving)
