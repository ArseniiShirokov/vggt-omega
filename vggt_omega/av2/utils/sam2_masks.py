from __future__ import annotations

import numpy as np

from vggt_omega.av2.utils.dynamic_boxes import clip_xyxy_to_image
from vggt_omega.av2.utils.lidar_prompts import DynamicObjectSamPrompt

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"


def load_sam2_predictor(model_id: str, device: str):
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "SAM 2 is required for dynamic mask filtering. Install it with `pip install sam2`."
        ) from exc

    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    if device != "cpu" and hasattr(predictor, "model"):
        predictor.model.to(device)
    return predictor


def _select_largest_mask_in_box(masks: np.ndarray, clipped: np.ndarray) -> np.ndarray:
    """Pick the SAM candidate with the largest area inside the prompt box."""
    x0, y0, x1, y1 = clipped.astype(int)
    areas = [mask[y0 : y1 + 1, x0 : x1 + 1].sum() for mask in masks]
    return masks[int(np.argmax(areas))]


def _clip_mask_to_box(mask: np.ndarray, clipped: np.ndarray) -> np.ndarray:
    """Zero out SAM mask pixels outside the prompt box."""
    x0, y0, x1, y1 = clipped.astype(int)
    clipped_mask = np.zeros(mask.shape, dtype=bool)
    clipped_mask[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return clipped_mask


def segment_dynamic_objects_sam2(
    image_rgb: np.ndarray,
    prompts: list[DynamicObjectSamPrompt],
    predictor,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    empty = np.zeros((height, width), dtype=np.uint8)
    if not prompts or predictor is None:
        return empty

    predictor.set_image(image_rgb)
    sam_union = np.zeros((height, width), dtype=bool)
    for prompt in prompts:
        clipped = clip_xyxy_to_image(
            prompt.xyxy[0],
            prompt.xyxy[1],
            prompt.xyxy[2],
            prompt.xyxy[3],
            width,
            height,
        )
        if clipped is None:
            continue

        point_coords = prompt.lidar_uv if len(prompt.lidar_uv) > 0 else None
        point_labels = (
            np.ones(len(prompt.lidar_uv), dtype=np.int32) if point_coords is not None else None
        )
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=clipped,
            multimask_output=True,
        )
        sam_union |= _clip_mask_to_box(
            _select_largest_mask_in_box(masks, clipped).astype(bool),
            clipped,
        )

    return sam_union.astype(np.uint8) * 255
