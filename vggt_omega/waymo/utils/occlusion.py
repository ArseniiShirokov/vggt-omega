from __future__ import annotations

import numpy as np

DEFAULT_OCCLUDER_NEGATIVE_POINTS = 8


def prompt_depth_cam(camera, prompt) -> float:
    """Camera-frame depth (z) of the 3D box center; smaller is closer."""
    center_ego = np.asarray(prompt.box.ego_SE3_object, dtype=np.float64)[:3, 3]
    cam_se3_ego = camera.ego_SE3_cam.inverse()
    return float(cam_se3_ego.transform_point_cloud(center_ego[None, :])[0, 2])


def is_box_fully_occluded(occluder_mask: np.ndarray, xyxy: np.ndarray) -> bool:
    """True when every pixel inside the projected 2D box is already masked."""
    x0, y0, x1, y1 = map(int, xyxy)
    region = occluder_mask[y0 : y1 + 1, x0 : x1 + 1]
    if region.size == 0:
        return False
    return bool(np.all(region))


def sample_occluder_negative_points(
    occluder_mask: np.ndarray,
    xyxy: np.ndarray,
    *,
    max_points: int = DEFAULT_OCCLUDER_NEGATIVE_POINTS,
    seed: int = 0,
) -> np.ndarray:
    """Sample negative SAM points from foreground mask pixels inside the prompt box."""
    x0, y0, x1, y1 = map(int, xyxy)
    patch = occluder_mask[y0 : y1 + 1, x0 : x1 + 1]
    ys, xs = np.where(patch)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    uv = np.column_stack([xs + x0, ys + y0]).astype(np.float32)
    if len(uv) <= max_points:
        return uv
    rng = np.random.default_rng(seed)
    return uv[rng.choice(len(uv), size=max_points, replace=False)]
