from __future__ import annotations

import numpy as np


def splat_points_in_camera(
    camera,
    points_cam: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Splat colored points into an image and a per-pixel depth (camera z, meters).

    Nearest points win per pixel (standard z-buffer). Points are sorted by pixel
    index, then by ascending depth so ``np.unique`` keeps the closest sample.
    """
    image = np.zeros((camera.height_px, camera.width_px, 3), dtype=np.uint8)
    depth_buffer = np.full((camera.height_px, camera.width_px), np.inf, dtype=np.float32)

    if len(points_cam) == 0:
        depth_buffer[:] = np.nan
        return image, depth_buffer

    uv, points_cam, valid = camera.project_cam_to_img(points_cam)
    uv = np.round(uv[valid]).astype(np.int32)
    z = points_cam[valid, 2]
    colors = colors[valid]

    u, v = uv[:, 0], uv[:, 1]
    in_bounds = (u >= 0) & (u < camera.width_px) & (v >= 0) & (v < camera.height_px) & (z > 0)
    u, v, z, colors = u[in_bounds], v[in_bounds], z[in_bounds], colors[in_bounds]
    if len(z) == 0:
        depth_buffer[:] = np.nan
        return image, depth_buffer

    lin_idx = v.astype(np.int64) * camera.width_px + u.astype(np.int64)
    order = np.lexsort((z, lin_idx))
    lin_sorted = lin_idx[order]
    _, nearest = np.unique(lin_sorted, return_index=True)

    u_nearest = u[order][nearest]
    v_nearest = v[order][nearest]
    z_nearest = z[order][nearest]
    colors_nearest = colors[order][nearest]

    image[v_nearest, u_nearest] = colors_nearest
    depth_buffer[v_nearest, u_nearest] = z_nearest
    depth_buffer[~np.isfinite(depth_buffer)] = np.nan
    return image, depth_buffer


def render_points_in_camera(camera, points_cam: np.ndarray, colors: np.ndarray) -> np.ndarray:
    image, _ = splat_points_in_camera(camera, points_cam, colors)
    return image
