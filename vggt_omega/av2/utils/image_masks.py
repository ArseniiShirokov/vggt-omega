from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from PIL import Image

from vggt_omega.av2.dataset import AV2Box3D
from vggt_omega.av2.utils.dynamic_boxes import clip_xyxy_to_image


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
    cropped_image, left, top = crop_native_for_vggt(image_rgb)
    crop_h, crop_w = cropped_image.shape[:2]
    cropped_mask = mask[top : top + crop_h, left : left + crop_w]
    return resize_mask(cropped_mask, pred_height, pred_width)


def apply_exclude_masks_to_conf(conf: np.ndarray, exclude_masks: list[np.ndarray]) -> np.ndarray:
    keep = np.array(exclude_masks) <= 0.1
    return conf * keep.astype(np.float32)


def shift_labeled_boxes(
    labeled_boxes: list[tuple[AV2Box3D, np.ndarray]],
    left: int,
    top: int,
) -> list[tuple[AV2Box3D, np.ndarray]]:
    if left == 0 and top == 0:
        return labeled_boxes
    offset = np.array([left, top, left, top], dtype=np.float32)
    return [(box, xyxy - offset) for box, xyxy in labeled_boxes]


def scale_labeled_boxes_to_pred(
    labeled_boxes: list[tuple[AV2Box3D, np.ndarray]],
    native_height: int,
    native_width: int,
    pred_height: int,
    pred_width: int,
) -> list[tuple[AV2Box3D, np.ndarray]]:
    scale_x = pred_width / native_width
    scale_y = pred_height / native_height
    scaled: list[tuple[AV2Box3D, np.ndarray]] = []
    for box, xyxy in labeled_boxes:
        clipped = clip_xyxy_to_image(
            xyxy[0] * scale_x,
            xyxy[1] * scale_y,
            xyxy[2] * scale_x,
            xyxy[3] * scale_y,
            pred_width,
            pred_height,
        )
        if clipped is not None:
            scaled.append((box, clipped))
    return scaled


def resize_native_to_pred_grid(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    labeled_boxes: list[tuple[AV2Box3D, np.ndarray]],
    *,
    pred_height: int,
    pred_width: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[AV2Box3D, np.ndarray]]]:
    cropped_image, left, top = crop_native_for_vggt(image_rgb)
    crop_h, crop_w = cropped_image.shape[:2]
    cropped_mask = mask[top : top + crop_h, left : left + crop_w]
    cropped_boxes = shift_labeled_boxes(labeled_boxes, left, top)
    resized_image = cv2.resize(
        cropped_image,
        (pred_width, pred_height),
        interpolation=cv2.INTER_AREA,
    )
    resized_mask = resize_mask(cropped_mask, pred_height, pred_width)
    resized_boxes = scale_labeled_boxes_to_pred(
        cropped_boxes,
        crop_h,
        crop_w,
        pred_height,
        pred_width,
    )
    return resized_image, resized_mask, resized_boxes
