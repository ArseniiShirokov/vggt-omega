from __future__ import annotations

import numpy as np

from vggt_omega.waymo.types import DYNAMIC_TYPE_IDS, WaymoBox3D, WaymoFrame
from vggt_omega.waymo.geometry import PinholeCamera, SE3

DEFAULT_BOX_FILTER_EXPAND_RATIO = 1.10
DEFAULT_MASK_CLIP_EXPAND_RATIO = 1.10
DEFAULT_SAM2_2D_EXPAND_RATIO = 1.25
DEFAULT_SAM2_2D_BOTTOM_EXPAND_RATIO = 1.0
DEFAULT_SAM2_2D_WIDTH_EXPAND_RATIO = 1.0
DEFAULT_WAYMO_MASK_CLOSE_ITERATIONS = 8
DEFAULT_WAYMO_MASK_CLOSE_KERNEL_SIZE = 5
DEFAULT_WAYMO_MASK_DILATE_ITERATIONS = 4
DEFAULT_WAYMO_MASK_DILATE_KERNEL_SIZE = 5


def waymo_sam_segment_kwargs() -> dict[str, object]:
    return {
        "sam_width_ratio": DEFAULT_SAM2_2D_WIDTH_EXPAND_RATIO,
        "sam_bottom_ratio": DEFAULT_SAM2_2D_BOTTOM_EXPAND_RATIO,
        "clip_to_prompt_box": False,
        "mask_close_iterations": DEFAULT_WAYMO_MASK_CLOSE_ITERATIONS,
        "mask_close_kernel_size": DEFAULT_WAYMO_MASK_CLOSE_KERNEL_SIZE,
        "mask_dilate_iterations": DEFAULT_WAYMO_MASK_DILATE_ITERATIONS,
        "mask_dilate_kernel_size": DEFAULT_WAYMO_MASK_DILATE_KERNEL_SIZE,
        "include_side_negatives": True,
        "mask_clip_expand_ratio": DEFAULT_MASK_CLIP_EXPAND_RATIO,
        "occlusion_aware": True,
    }


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
    frame: WaymoFrame,
    *,
    moving_tracks: frozenset[str] | None = None,
) -> tuple[WaymoBox3D, ...]:
    boxes = tuple(box for box in frame.boxes if box.type_id in DYNAMIC_TYPE_IDS)
    if moving_tracks is None:
        return boxes
    return tuple(box for box in boxes if box.track_id in moving_tracks)


def box_vertices_ego(box: WaymoBox3D, *, expand_ratio: float = 1.0) -> np.ndarray:
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
    ego_se3_object = SE3.from_matrix(box.ego_SE3_object)
    half = np.array([box.length_m, box.width_m, box.height_m], dtype=float) / 2.0 * expand_ratio
    return ego_se3_object.transform_point_cloud(half * unit_vertices)


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
    cx = (xyxy[0] + xyxy[2]) * 0.5
    cy = (xyxy[1] + xyxy[3]) * 0.5
    w = (xyxy[2] - xyxy[0]) * expand_ratio
    h = (xyxy[3] - xyxy[1]) * expand_ratio
    return clip_xyxy_to_image(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5, width, height)


def expand_xyxy_for_sam(
    xyxy: np.ndarray,
    width: int,
    height: int,
    *,
    width_ratio: float = DEFAULT_SAM2_2D_EXPAND_RATIO,
    bottom_ratio: float = DEFAULT_SAM2_2D_BOTTOM_EXPAND_RATIO,
) -> np.ndarray | None:
    x0, y0, x1, y1 = map(float, xyxy)
    cx = (x0 + x1) * 0.5
    w = (x1 - x0) * width_ratio
    y1 = y1 + (y1 - y0) * (bottom_ratio - 1.0)
    return clip_xyxy_to_image(cx - w * 0.5, y0, cx + w * 0.5, y1, width, height)


def project_box_to_xyxy(
    box: WaymoBox3D,
    camera: PinholeCamera,
    *,
    expand_ratio: float = 1.0,
) -> np.ndarray | None:
    uv, points_cam, valid = camera.project_ego_to_img(box_vertices_ego(box, expand_ratio=expand_ratio))
    in_front = valid & (points_cam[:, 2] > 0)
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
    frame: WaymoFrame,
    camera: PinholeCamera,
    *,
    moving_tracks: frozenset[str] | None = None,
    expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
) -> list[tuple[WaymoBox3D, np.ndarray]]:
    labeled: list[tuple[WaymoBox3D, np.ndarray]] = []
    for box in dynamic_boxes(frame, moving_tracks=moving_tracks):
        xyxy = project_box_to_xyxy(box, camera, expand_ratio=expand_ratio)
        if xyxy is not None:
            labeled.append((box, xyxy))
    return labeled


def points_inside_box(
    points_ego: np.ndarray,
    box: WaymoBox3D,
    *,
    expand_ratio: float = 1.0,
) -> np.ndarray:
    ego_se3_object = SE3.from_matrix(box.ego_SE3_object)
    points_obj = ego_se3_object.inverse().transform_point_cloud(points_ego)
    half = np.array([box.length_m, box.width_m, box.height_m], dtype=float) * 0.5 * expand_ratio
    return (
        (np.abs(points_obj[:, 0]) <= half[0])
        & (np.abs(points_obj[:, 1]) <= half[1])
        & (np.abs(points_obj[:, 2]) <= half[2])
    )


def box_wireframe_edges() -> tuple[tuple[int, int], ...]:
    return _BOX_EDGES
