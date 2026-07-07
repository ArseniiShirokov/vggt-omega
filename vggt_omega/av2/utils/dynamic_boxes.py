from __future__ import annotations

import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3

from vggt_omega.av2.dataset import AV2Box3D, AV2Frame

DYNAMIC_CATEGORIES = frozenset(
    {
        "ANIMAL",
        "ARTICULATED_BUS",
        "BICYCLE",
        "BICYCLIST",
        "BOX_TRUCK",
        "BUS",
        "DOG",
        "LARGE_VEHICLE",
        "MESSAGE_BOARD_TRAILER",
        "MOTORCYCLE",
        "MOTORCYCLIST",
        "OFFICIAL_SIGNALER",
        "PEDESTRIAN",
        "RAILED_VEHICLE",
        "REGULAR_VEHICLE",
        "SCHOOL_BUS",
        "STROLLER",
        "TRAFFIC_LIGHT_TRAILER",
        "TRUCK",
        "TRUCK_CAB",
        "VEHICULAR_TRAILER",
        "WHEELCHAIR",
        "WHEELED_DEVICE",
        "WHEELED_RIDER",
    }
)

DEFAULT_BOX_FILTER_EXPAND_RATIO = 1.15

_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def dynamic_boxes(
    frame: AV2Frame,
    *,
    moving_tracks: frozenset[str] | None = None,
) -> tuple[AV2Box3D, ...]:
    boxes = tuple(box for box in frame.boxes if box.category in DYNAMIC_CATEGORIES)
    if moving_tracks is None:
        return boxes
    return tuple(box for box in boxes if box.track_uuid in moving_tracks)


def box_vertices_ego(box: AV2Box3D, *, expand_ratio: float = 1.0) -> np.ndarray:
    unit_vertices = np.array(
        [
            [+1, +1, +1],
            [+1, -1, +1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, +1],
            [-1, -1, +1],
            [-1, -1, -1],
            [-1, +1, -1],
        ],
        dtype=float,
    )
    ego_se3_object = SE3(
        rotation=box.ego_SE3_object[:3, :3],
        translation=box.ego_SE3_object[:3, 3],
    )
    half = (np.array([box.length_m, box.width_m, box.height_m], dtype=float) / 2.0) * expand_ratio
    vertices_obj = half * unit_vertices
    return ego_se3_object.transform_point_cloud(vertices_obj)


def clip_xyxy_to_image(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    x0 = int(np.floor(x0))
    y0 = int(np.floor(y0))
    x1 = int(np.ceil(x1))
    y1 = int(np.ceil(y1))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def expand_xyxy(
    xyxy: np.ndarray,
    expand_ratio: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Scale a 2D box around its center, then clip to the image."""
    cx = (xyxy[0] + xyxy[2]) * 0.5
    cy = (xyxy[1] + xyxy[3]) * 0.5
    w = (xyxy[2] - xyxy[0]) * expand_ratio
    h = (xyxy[3] - xyxy[1]) * expand_ratio
    return clip_xyxy_to_image(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5, width, height)


DEFAULT_SAM2_2D_EXPAND_RATIO = 1.25


def expand_xyxy_for_sam(
    xyxy: np.ndarray,
    width: int,
    height: int,
    *,
    width_ratio: float = DEFAULT_SAM2_2D_EXPAND_RATIO,
    bottom_ratio: float = DEFAULT_SAM2_2D_EXPAND_RATIO,
) -> np.ndarray | None:
    """Widen and extend the bottom of a projected 2D box (wheels sit below AV2 3D boxes)."""
    x0, y0, x1, y1 = map(float, xyxy)
    cx = (x0 + x1) * 0.5
    w = (x1 - x0) * width_ratio
    y1 = y1 + (y1 - y0) * (bottom_ratio - 1.0)
    return clip_xyxy_to_image(cx - w * 0.5, y0, cx + w * 0.5, y1, width, height)


def project_box_to_xyxy(
    box: AV2Box3D,
    camera: PinholeCamera,
    *,
    expand_ratio: float = 1.0,
) -> np.ndarray | None:
    uv, points_cam, _ = camera.project_ego_to_img(box_vertices_ego(box, expand_ratio=expand_ratio))
    in_front = points_cam[:, 2] > 0
    if not np.any(in_front):
        return None
    uv = uv[in_front]
    return clip_xyxy_to_image(
        uv[:, 0].min(),
        uv[:, 1].min(),
        uv[:, 0].max(),
        uv[:, 1].max(),
        camera.width_px,
        camera.height_px,
    )


def project_dynamic_boxes_labeled(
    frame: AV2Frame,
    camera: PinholeCamera,
    *,
    moving_tracks: frozenset[str] | None = None,
    expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
) -> list[tuple[AV2Box3D, np.ndarray]]:
    labeled: list[tuple[AV2Box3D, np.ndarray]] = []
    for box in dynamic_boxes(frame, moving_tracks=moving_tracks):
        xyxy = project_box_to_xyxy(box, camera, expand_ratio=expand_ratio)
        if xyxy is not None:
            labeled.append((box, xyxy))
    return labeled


def points_inside_box(
    points_ego: np.ndarray,
    box: AV2Box3D,
    *,
    expand_ratio: float = 1.0,
) -> np.ndarray:
    ego_se3_object = SE3(
        rotation=box.ego_SE3_object[:3, :3],
        translation=box.ego_SE3_object[:3, 3],
    )
    points_obj = ego_se3_object.inverse().transform_point_cloud(points_ego)
    half = np.array([box.length_m, box.width_m, box.height_m], dtype=float) * 0.5 * expand_ratio
    return (
        (np.abs(points_obj[:, 0]) <= half[0])
        & (np.abs(points_obj[:, 1]) <= half[1])
        & (np.abs(points_obj[:, 2]) <= half[2])
    )


def box_wireframe_edges() -> tuple[tuple[int, int], ...]:
    return _BOX_EDGES
