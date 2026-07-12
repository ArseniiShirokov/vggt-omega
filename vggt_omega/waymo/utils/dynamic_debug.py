from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vggt_omega.waymo.geometry import PinholeCamera
from vggt_omega.waymo.types import WaymoBox3D, WaymoFrame
from vggt_omega.waymo.utils.dynamic_boxes import (
    DEFAULT_BOX_FILTER_EXPAND_RATIO,
    DEFAULT_MASK_CLIP_EXPAND_RATIO,
    box_vertices_ego,
    box_wireframe_edges,
    dynamic_boxes,
)
from vggt_omega.waymo.utils.image_masks import mask_to_pred_grid, resize_native_to_pred_grid
from vggt_omega.waymo.utils.lidar_prompts import DynamicObjectSamPrompt, load_frame_image_rgb

STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("raw", "01 SAM raw"),
    ("fill_holes", "02 fill holes"),
    ("closing", "03 closing"),
    ("dilated", "04 dilate"),
    ("postprocessed", "05 postprocessed"),
    ("clipped", "06 clip box"),
    ("final", "07 final union"),
)


def _draw_xyxy(
    image_bgr: np.ndarray,
    xyxy: np.ndarray,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    label: str | None = None,
) -> None:
    x0, y0, x1, y1 = map(int, xyxy)
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(
            image_bgr,
            label,
            (x0, max(y0 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_prompts_debug(
    image_rgb: np.ndarray,
    prompts: list[DynamicObjectSamPrompt],
    *,
    frame: WaymoFrame | None = None,
    camera: PinholeCamera | None = None,
    per_object_debug: list[dict[str, object]] | None = None,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    if frame is not None and camera is not None:
        for box in dynamic_boxes(frame):
            uv, points_cam, valid = camera.project_ego_to_img(
                box_vertices_ego(box, expand_ratio=DEFAULT_BOX_FILTER_EXPAND_RATIO)
            )
            for i, j in box_wireframe_edges():
                if not (valid[i] and valid[j] and points_cam[i, 2] > 0 and points_cam[j, 2] > 0):
                    continue
                p0 = tuple(np.round(uv[i]).astype(int))
                p1 = tuple(np.round(uv[j]).astype(int))
                cv2.line(image_bgr, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)

    debug_by_track = {
        getattr(item["prompt"], "box").track_id: item for item in (per_object_debug or [])
    }

    for prompt in prompts:
        _draw_xyxy(image_bgr, prompt.xyxy, (0, 255, 0), label=prompt.category)

        item = debug_by_track.get(prompt.box.track_id)
        if item is not None:
            boxes = item["boxes"]
            _draw_xyxy(image_bgr, boxes["sam_box"], (255, 255, 0), thickness=1, label="SAM box")
            _draw_xyxy(image_bgr, boxes["mask_clip_box"], (255, 0, 255), thickness=1, label="clip box")

            point_coords = boxes["point_coords"]
            point_labels = boxes["point_labels"]
            for uv, label in zip(point_coords, point_labels, strict=True):
                color = (0, 255, 255) if label == 1 else (0, 0, 255)
                cv2.circle(
                    image_bgr,
                    tuple(np.round(uv).astype(int)),
                    4 if label == 1 else 3,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            occluder_neg = boxes.get("occluder_negative_uv")
            if occluder_neg is not None and len(occluder_neg) > 0:
                for uv in occluder_neg:
                    cv2.circle(
                        image_bgr,
                        tuple(np.round(uv).astype(int)),
                        4,
                        (255, 0, 255),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
        else:
            for uv in prompt.lidar_uv:
                cv2.circle(
                    image_bgr,
                    tuple(np.round(uv).astype(int)),
                    4,
                    (0, 255, 255),
                    -1,
                    lineType=cv2.LINE_AA,
                )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def overlay_mask_debug(
    image_rgb: np.ndarray,
    mask: np.ndarray | np.bool_,
    *,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.copy()
    active = np.asarray(mask) > 0
    if not np.any(active):
        return overlay

    tint = np.zeros_like(image_rgb)
    tint[active] = color
    overlay[active] = (
        alpha * tint[active].astype(np.float32) + (1.0 - alpha) * overlay[active].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def _mask_to_vis(mask: np.ndarray | np.bool_) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _label_image(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    if labeled.ndim == 2:
        labeled = cv2.cvtColor(labeled, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        labeled,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        labeled,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return labeled


def make_stage_panel(
    image_rgb: np.ndarray,
    stage_unions: dict[str, np.ndarray],
    *,
    prompts_vis: np.ndarray | None = None,
) -> np.ndarray:
    tiles: list[np.ndarray] = []
    if prompts_vis is not None:
        tiles.append(_label_image(cv2.cvtColor(prompts_vis, cv2.COLOR_RGB2BGR), "00 prompts"))

    for key, label in STAGE_LABELS:
        if key not in stage_unions:
            continue
        overlay = overlay_mask_debug(image_rgb, stage_unions[key])
        tiles.append(_label_image(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), label))

    if not tiles:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    tile_h, tile_w = tiles[0].shape[:2]
    resized = [cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for tile in tiles]
    cols = 3
    rows = int(np.ceil(len(resized) / cols))
    while len(resized) < rows * cols:
        resized.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    row_images = []
    for row in range(rows):
        row_images.append(np.hstack(resized[row * cols : (row + 1) * cols]))
    return np.vstack(row_images)


def save_dynamic_filter_debug(
    frames: list[WaymoFrame],
    image_paths: list[Path | str],
    camera: PinholeCamera,
    output_dir: str | Path,
    *,
    prompts_per_frame: list[list[DynamicObjectSamPrompt]],
    masks_per_frame: list[np.ndarray],
    debug_info_per_frame: list[dict[str, object]] | None = None,
    crop_bottom: int = 0,
    pred_height: int | None = None,
    pred_width: int | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for frame_idx, (frame, image_path, prompts, mask) in enumerate(
        zip(frames, image_paths, prompts_per_frame, masks_per_frame, strict=True)
    ):
        prefix = f"{frame.timestamp_us}"
        frame_dir = output_dir / prefix
        frame_dir.mkdir(parents=True, exist_ok=True)

        image_rgb = load_frame_image_rgb(image_path, crop_bottom=crop_bottom)
        debug_info = (
            debug_info_per_frame[frame_idx]
            if debug_info_per_frame is not None and frame_idx < len(debug_info_per_frame)
            else None
        )
        per_object = [] if debug_info is None else debug_info.get("per_object", [])
        stage_unions = {} if debug_info is None else debug_info.get("stage_unions", {})

        prompts_vis = draw_prompts_debug(
            image_rgb,
            prompts,
            frame=frame,
            camera=camera,
            per_object_debug=per_object,
        )
        overlay_vis = overlay_mask_debug(image_rgb, mask)
        combined_vis = overlay_mask_debug(prompts_vis, mask)

        paths = {
            "prompts": frame_dir / "00_prompts.jpg",
            "mask": frame_dir / "mask.png",
            "overlay": frame_dir / "overlay.jpg",
            "combined": frame_dir / "combined.jpg",
        }
        cv2.imwrite(str(paths["prompts"]), cv2.cvtColor(prompts_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(paths["mask"]), mask)
        cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(overlay_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(paths["combined"]), cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
        saved_paths.extend(paths.values())

        for key, label in STAGE_LABELS:
            if key not in stage_unions:
                continue
            slug = label.lower().replace(" ", "_")
            stage_mask = _mask_to_vis(stage_unions[key])
            stage_overlay = overlay_mask_debug(image_rgb, stage_mask)
            stage_path = frame_dir / f"{slug}.jpg"
            mask_only_path = frame_dir / f"{slug}.png"
            cv2.imwrite(str(stage_path), cv2.cvtColor(stage_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_only_path), stage_mask)
            saved_paths.extend([stage_path, mask_only_path])

        panel = make_stage_panel(image_rgb, stage_unions, prompts_vis=prompts_vis)
        panel_path = frame_dir / "panel.jpg"
        cv2.imwrite(str(panel_path), panel)
        saved_paths.append(panel_path)

        if pred_height is not None and pred_width is not None:
            labeled_boxes = [(prompt.box, prompt.xyxy) for prompt in prompts]
            resized_image, resized_mask, _ = resize_native_to_pred_grid(
                image_rgb,
                mask,
                labeled_boxes,
                pred_height=pred_height,
                pred_width=pred_width,
            )
            mask_grid = mask_to_pred_grid(
                image_rgb,
                mask,
                pred_height=pred_height,
                pred_width=pred_width,
            )
            resized_overlay = overlay_mask_debug(resized_image, resized_mask)
            grid_overlay = overlay_mask_debug(
                cv2.resize(image_rgb, (pred_width, pred_height), interpolation=cv2.INTER_AREA),
                (mask_grid > 0.1).astype(np.uint8) * 255,
            )
            resized_path = frame_dir / f"overlay_{pred_height}x{pred_width}.jpg"
            grid_path = frame_dir / f"overlay_pred_grid_{pred_height}x{pred_width}.jpg"
            cv2.imwrite(str(resized_path), cv2.cvtColor(resized_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(grid_path), cv2.cvtColor(grid_overlay, cv2.COLOR_RGB2BGR))
            saved_paths.extend([resized_path, grid_path])

    return saved_paths
