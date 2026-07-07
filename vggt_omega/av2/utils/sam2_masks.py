from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure

from vggt_omega.av2.utils.dynamic_boxes import clip_xyxy_to_image, expand_xyxy_for_sam
from vggt_omega.av2.utils.lidar_prompts import DynamicObjectSamPrompt

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-base-plus"
DEFAULT_NEGATIVE_CORNER_INSET = 0.08
DEFAULT_MASK_CLOSE_ITERATIONS = 2


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


def _negative_box_corner_points(clipped: np.ndarray, *, inset: float = DEFAULT_NEGATIVE_CORNER_INSET) -> np.ndarray:
    """Background prompts near box corners (red points) to suppress road/sky bleed."""
    x0, y0, x1, y1 = map(float, clipped)
    dx = (x1 - x0) * inset
    dy = (y1 - y0) * inset
    return np.array(
        [
            [x0 + dx, y0 + dy],
            [x1 - dx, y0 + dy],
            [x0 + dx, y1 - dy],
            [x1 - dx, y1 - dy],
        ],
        dtype=np.float32,
    )


def _build_sam_point_prompts(lidar_uv: np.ndarray, clipped: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    negatives = _negative_box_corner_points(clipped)
    if len(lidar_uv) == 0:
        return None, None

    positives = lidar_uv.astype(np.float32)
    point_coords = np.vstack([positives, negatives])
    point_labels = np.concatenate(
        [
            np.ones(len(positives), dtype=np.int32),
            np.zeros(len(negatives), dtype=np.int32),
        ]
    )
    return point_coords, point_labels


def _select_best_multimask(masks: np.ndarray, iou_scores: np.ndarray) -> np.ndarray:
    """Pick the multimask candidate with the highest SAM IoU score."""
    iou_scores = np.asarray(iou_scores, dtype=np.float64).reshape(-1)
    return masks[int(np.argmax(iou_scores))]


def _postprocess_mask(mask: np.ndarray, clipped: np.ndarray) -> np.ndarray:
    """Fill small holes and close thin gaps inside the prompt box."""
    x0, y0, x1, y1 = clipped.astype(int)
    roi = mask[y0 : y1 + 1, x0 : x1 + 1].copy()
    roi = binary_fill_holes(roi)
    if DEFAULT_MASK_CLOSE_ITERATIONS > 0:
        structure = generate_binary_structure(2, 1)
        roi = binary_closing(roi, structure=structure, iterations=DEFAULT_MASK_CLOSE_ITERATIONS)
    result = mask.copy()
    result[y0 : y1 + 1, x0 : x1 + 1] = roi
    return result


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
        clipped = expand_xyxy_for_sam(clipped, width, height)
        if clipped is None:
            continue

        point_coords, point_labels = _build_sam_point_prompts(prompt.lidar_uv, clipped)
        masks, iou_scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=clipped,
            multimask_output=True,
        )
        selected = _select_best_multimask(masks, iou_scores).astype(bool)
        selected = _postprocess_mask(selected, clipped)
        sam_union |= _clip_mask_to_box(selected, clipped)

    return sam_union.astype(np.uint8) * 255
