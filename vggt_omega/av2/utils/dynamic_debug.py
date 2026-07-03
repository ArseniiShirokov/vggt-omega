from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera

from vggt_omega.av2.dataset import AV2Box3D, AV2Frame
from vggt_omega.av2.utils.dynamic_boxes import (
    DEFAULT_BOX_FILTER_EXPAND_RATIO,
    box_vertices_ego,
    box_wireframe_edges,
    dynamic_boxes,
)
from vggt_omega.av2.utils.image_masks import resize_native_to_pred_grid
from vggt_omega.av2.utils.lidar_prompts import DynamicObjectSamPrompt


def prompts_to_labeled_boxes(
    prompts: list[DynamicObjectSamPrompt],
) -> list[tuple[AV2Box3D, np.ndarray]]:
    return [(prompt.box, prompt.xyxy) for prompt in prompts]


def draw_projected_boxes_debug(
    image_rgb: np.ndarray,
    prompts: list[DynamicObjectSamPrompt],
    *,
    frame: AV2Frame | None = None,
    camera: PinholeCamera | None = None,
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

    for prompt in prompts:
        x0, y0, x1, y1 = map(int, prompt.xyxy)
        color = (0, 255, 0) if prompt.use_3d_box else (0, 128, 255)
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        for uv in prompt.lidar_uv:
            cv2.circle(
                image_bgr,
                tuple(np.round(uv).astype(int)),
                3,
                (255, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )
        label = prompt.category
        if prompt.scale_error is not None:
            tags = ["SAM"]
            if prompt.use_3d_box:
                tags.append("3D")
            label = f"{prompt.category} [{'+'.join(tags)} err={prompt.scale_error:.2f}]"
        elif len(prompt.lidar_uv) > 0:
            label = f"{prompt.category} ({len(prompt.lidar_uv)} pts)"
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

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def overlay_dynamic_mask_debug(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    prompts: list[DynamicObjectSamPrompt] | None = None,
    labeled_boxes: list[tuple[AV2Box3D, np.ndarray]] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    if prompts is not None:
        labeled_boxes = prompts_to_labeled_boxes(prompts)

    overlay = image_rgb.copy()
    dynamic = mask > 0
    if not np.any(dynamic):
        return overlay

    tint = np.zeros_like(image_rgb)
    tint[dynamic] = (255, 64, 64)
    overlay[dynamic] = (
        alpha * tint[dynamic].astype(np.float32) + (1.0 - alpha) * overlay[dynamic].astype(np.float32)
    ).astype(np.uint8)

    for box, xyxy in labeled_boxes or []:
        x0, y0, x1, y1 = map(int, xyxy)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 64, 64), 2)

    if prompts is not None:
        for prompt in prompts:
            for uv in prompt.lidar_uv:
                cv2.circle(
                    overlay,
                    tuple(np.round(uv).astype(int)),
                    3,
                    (255, 255, 0),
                    -1,
                    lineType=cv2.LINE_AA,
                )

    return overlay


def save_dynamic_filter_debug(
    frames: list[AV2Frame],
    image_paths: list[Path | str],
    camera: PinholeCamera,
    output_dir: str | Path,
    *,
    prompts_per_frame: list[list[DynamicObjectSamPrompt]],
    masks_per_frame: list[np.ndarray],
    crop_bottom: int = 0,
    pred_height: int | None = None,
    pred_width: int | None = None,
    load_image,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for frame, image_path, prompts, mask in zip(
        frames, image_paths, prompts_per_frame, masks_per_frame, strict=True
    ):
        prefix = f"{frame.cam_timestamp_ns:020d}"
        image_rgb = load_image(image_path, crop_bottom=crop_bottom)
        labeled_boxes = prompts_to_labeled_boxes(prompts)

        boxes_vis = draw_projected_boxes_debug(image_rgb, prompts, frame=frame, camera=camera)
        overlay_vis = overlay_dynamic_mask_debug(image_rgb, mask, prompts=prompts)
        combined_vis = overlay_dynamic_mask_debug(boxes_vis, mask, prompts=prompts)

        paths = {
            "boxes": output_dir / f"{prefix}_boxes.jpg",
            "sam_mask": output_dir / f"{prefix}_sam_mask.png",
            "mask": output_dir / f"{prefix}_mask.png",
            "overlay": output_dir / f"{prefix}_overlay.jpg",
            "combined": output_dir / f"{prefix}_combined.jpg",
        }
        cv2.imwrite(str(paths["boxes"]), cv2.cvtColor(boxes_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(paths["sam_mask"]), mask)
        cv2.imwrite(str(paths["mask"]), mask)
        cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(overlay_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(paths["combined"]), cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
        saved_paths.extend(paths.values())

        if pred_height is not None and pred_width is not None:
            resized_image, resized_mask, resized_boxes = resize_native_to_pred_grid(
                image_rgb,
                mask,
                labeled_boxes,
                pred_height=pred_height,
                pred_width=pred_width,
            )
            resized_overlay = overlay_dynamic_mask_debug(
                resized_image,
                resized_mask,
                labeled_boxes=resized_boxes,
            )
            resized_path = output_dir / f"{prefix}_overlay_{pred_height}x{pred_width}.jpg"
            cv2.imwrite(str(resized_path), cv2.cvtColor(resized_overlay, cv2.COLOR_RGB2BGR))
            saved_paths.append(resized_path)

    return saved_paths
