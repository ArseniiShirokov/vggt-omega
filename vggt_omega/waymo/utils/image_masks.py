from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vggt_omega.waymo.types import WaymoBox3D


def resize_mask(mask: np.ndarray, pred_height: int, pred_width: int) -> np.ndarray:
    if mask.shape[0] == pred_height and mask.shape[1] == pred_width:
        return mask
    return cv2.resize(mask, (pred_width, pred_height), interpolation=cv2.INTER_NEAREST)


def crop_native_for_vggt(
    image_rgb: np.ndarray,
    *,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[np.ndarray, int, int]:
    native_h, native_w = image_rgb.shape[:2]
    aspect_ratio = native_h / max(native_w, 1)
    pil = Image.fromarray(image_rgb)

    if aspect_ratio < min_aspect_ratio:
        crop_w = min(native_w, max(1, int(round(native_h / min_aspect_ratio))))
        left = max((native_w - crop_w) // 2, 0)
        return np.array(pil.crop((left, 0, left + crop_w, native_h))), left, 0

    if aspect_ratio > max_aspect_ratio:
        crop_h = min(native_h, max(1, int(round(native_w * max_aspect_ratio))))
        top = max((native_h - crop_h) // 2, 0)
        return np.array(pil.crop((0, top, native_w, top + crop_h))), 0, top

    return image_rgb, 0, 0


def mask_to_pred_grid(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    pred_height: int,
    pred_width: int,
) -> np.ndarray:
    cropped_rgb, crop_left, crop_top = crop_native_for_vggt(image_rgb)
    cropped_h, cropped_w = cropped_rgb.shape[:2]
    scale = min(pred_width / cropped_w, pred_height / cropped_h)
    resized_w = max(1, int(round(cropped_w * scale)))
    resized_h = max(1, int(round(cropped_h * scale)))

    pad_left = max((pred_width - resized_w) // 2, 0)
    pad_top = max((pred_height - resized_h) // 2, 0)

    mask_cropped = mask[crop_top : crop_top + cropped_h, crop_left : crop_left + cropped_w]
    mask_resized = resize_mask(mask_cropped, resized_h, resized_w)

    grid = np.zeros((pred_height, pred_width), dtype=np.float32)
    grid[
        pad_top : pad_top + resized_h,
        pad_left : pad_left + resized_w,
    ] = mask_resized.astype(np.float32) / 255.0
    return grid


def apply_exclude_masks_to_conf(
    conf: np.ndarray,
    exclude_masks: list[np.ndarray],
) -> np.ndarray:
    filtered = conf.copy()
    for index, exclude in enumerate(exclude_masks):
        filtered[index] = filtered[index] * (1.0 - exclude)
    return filtered


def resize_native_to_pred_grid(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    labeled_boxes: list[tuple[WaymoBox3D, np.ndarray]],
    *,
    pred_height: int,
    pred_width: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[WaymoBox3D, np.ndarray]]]:
    cropped_rgb, crop_left, crop_top = crop_native_for_vggt(image_rgb)
    cropped_h, cropped_w = cropped_rgb.shape[:2]
    scale = min(pred_width / cropped_w, pred_height / cropped_h)
    resized_w = max(1, int(round(cropped_w * scale)))
    resized_h = max(1, int(round(cropped_h * scale)))
    pad_left = max((pred_width - resized_w) // 2, 0)
    pad_top = max((pred_height - resized_h) // 2, 0)

    image_resized = cv2.resize(cropped_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    image_grid = np.zeros((pred_height, pred_width, 3), dtype=np.uint8)
    image_grid[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = image_resized

    mask_cropped = mask[crop_top : crop_top + cropped_h, crop_left : crop_left + cropped_w]
    mask_resized = resize_mask(mask_cropped, resized_h, resized_w)
    mask_grid = np.zeros((pred_height, pred_width), dtype=np.uint8)
    mask_grid[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = mask_resized

    def _map_xyxy(xyxy: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = xyxy
        x0 = (x0 - crop_left) * scale + pad_left
        x1 = (x1 - crop_left) * scale + pad_left
        y0 = (y0 - crop_top) * scale + pad_top
        y1 = (y1 - crop_top) * scale + pad_top
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    resized_boxes = [(box, _map_xyxy(xyxy)) for box, xyxy in labeled_boxes]
    return image_grid, mask_grid, resized_boxes
