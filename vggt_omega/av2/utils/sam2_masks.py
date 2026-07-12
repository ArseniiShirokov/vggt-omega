from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, generate_binary_structure

from vggt_omega.av2.utils.dynamic_boxes import clip_xyxy_to_image, expand_xyxy, expand_xyxy_for_sam
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
    if hasattr(predictor, "model"):
        predictor.model.eval()
        if device != "cpu":
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


def _negative_box_side_points(
    clipped: np.ndarray,
    *,
    inset: float = DEFAULT_NEGATIVE_CORNER_INSET,
) -> np.ndarray:
    """Background prompts on left/right box edges to suppress lateral bleed."""
    x0, y0, x1, y1 = map(float, clipped)
    dx = (x1 - x0) * inset
    dy = (y1 - y0) * inset
    cy = (y0 + y1) * 0.5
    return np.array(
        [
            [x0 + dx, cy],
            [x1 - dx, cy],
            [x0 + dx, y0 + dy],
            [x1 - dx, y0 + dy],
            [x0 + dx, y1 - dy],
            [x1 - dx, y1 - dy],
        ],
        dtype=np.float32,
    )


def _build_sam_point_prompts(
    lidar_uv: np.ndarray,
    clipped: np.ndarray,
    *,
    include_side_negatives: bool = False,
    occluder_negative_uv: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    negatives = _negative_box_corner_points(clipped)
    if include_side_negatives:
        negatives = np.vstack([negatives, _negative_box_side_points(clipped)])
    if occluder_negative_uv is not None and len(occluder_negative_uv) > 0:
        negatives = np.vstack([negatives, occluder_negative_uv.astype(np.float32)])

    if len(lidar_uv) == 0:
        return negatives.astype(np.float32), np.zeros(len(negatives), dtype=np.int32)

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


def _binary_structure(kernel_size: int) -> np.ndarray:
    if kernel_size > 1:
        return np.ones((kernel_size, kernel_size), dtype=bool)
    return generate_binary_structure(2, 1)


def _postprocess_mask_stages(
    mask: np.ndarray,
    clipped: np.ndarray,
    *,
    close_iterations: int = DEFAULT_MASK_CLOSE_ITERATIONS,
    close_kernel_size: int = 1,
    dilate_iterations: int = 0,
    dilate_kernel_size: int = 1,
) -> dict[str, np.ndarray]:
    """Expand the SAM mask monotonically: each step only adds pixels."""
    del clipped  # postprocess runs on the full mask, not a cropped ROI
    current = mask.astype(bool).copy()
    stages: dict[str, np.ndarray] = {"raw": current.copy()}

    current |= binary_fill_holes(current)
    stages["fill_holes"] = current.copy()

    if close_iterations > 0:
        current |= binary_closing(
            current,
            structure=_binary_structure(close_kernel_size),
            iterations=close_iterations,
        )
    stages["closing"] = current.copy()

    if dilate_iterations > 0:
        current = binary_dilation(
            current,
            structure=_binary_structure(dilate_kernel_size),
            iterations=dilate_iterations,
        )
    stages["dilated"] = current.copy()

    stages["postprocessed"] = current.copy()
    return stages


def _postprocess_mask(
    mask: np.ndarray,
    clipped: np.ndarray,
    *,
    close_iterations: int = DEFAULT_MASK_CLOSE_ITERATIONS,
    close_kernel_size: int = 1,
    dilate_iterations: int = 0,
    dilate_kernel_size: int = 1,
    keep_largest_component: bool = False,
) -> np.ndarray:
    """Fill enclosed holes and close thin gaps without shrinking the SAM mask."""
    del keep_largest_component
    return _postprocess_mask_stages(
        mask,
        clipped,
        close_iterations=close_iterations,
        close_kernel_size=close_kernel_size,
        dilate_iterations=dilate_iterations,
        dilate_kernel_size=dilate_kernel_size,
    )["postprocessed"]


def _resolve_sam_boxes(
    prompt,
    width: int,
    height: int,
    *,
    sam_width_ratio: float | None,
    sam_bottom_ratio: float | None,
    clip_to_prompt_box: bool,
    mask_clip_expand_ratio: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    tight_box = clip_xyxy_to_image(
        prompt.xyxy[0],
        prompt.xyxy[1],
        prompt.xyxy[2],
        prompt.xyxy[3],
        width,
        height,
    )
    if tight_box is None:
        return None, None, None

    clipped = tight_box
    if sam_width_ratio is None and sam_bottom_ratio is None:
        clipped = expand_xyxy_for_sam(clipped, width, height)
    else:
        expand_kwargs: dict[str, float] = {}
        if sam_width_ratio is not None:
            expand_kwargs["width_ratio"] = sam_width_ratio
        if sam_bottom_ratio is not None:
            expand_kwargs["bottom_ratio"] = sam_bottom_ratio
        clipped = expand_xyxy_for_sam(clipped, width, height, **expand_kwargs)
    if clipped is None:
        return tight_box, None, None

    mask_clip_box = tight_box if clip_to_prompt_box else clipped
    if clip_to_prompt_box and mask_clip_expand_ratio > 1.0:
        expanded_clip = expand_xyxy(mask_clip_box, mask_clip_expand_ratio, width, height)
        if expanded_clip is not None:
            mask_clip_box = expanded_clip
    return tight_box, clipped, mask_clip_box


def _segment_prompt_mask(
    prompt,
    predictor,
    width: int,
    height: int,
    *,
    sam_width_ratio: float | None,
    sam_bottom_ratio: float | None,
    clip_to_prompt_box: bool,
    mask_close_iterations: int | None,
    mask_close_kernel_size: int,
    mask_dilate_iterations: int,
    mask_dilate_kernel_size: int,
    keep_largest_component: bool,
    include_side_negatives: bool,
    mask_clip_expand_ratio: float,
    occluder_negative_uv: np.ndarray | None = None,
    collect_stages: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    tight_box, clipped, mask_clip_box = _resolve_sam_boxes(
        prompt,
        width,
        height,
        sam_width_ratio=sam_width_ratio,
        sam_bottom_ratio=sam_bottom_ratio,
        clip_to_prompt_box=clip_to_prompt_box,
        mask_clip_expand_ratio=mask_clip_expand_ratio,
    )
    if clipped is None or mask_clip_box is None:
        return np.zeros((height, width), dtype=np.uint8), None, None

    point_coords, point_labels = _build_sam_point_prompts(
        prompt.lidar_uv,
        clipped,
        include_side_negatives=include_side_negatives,
        occluder_negative_uv=occluder_negative_uv,
    )
    with torch.inference_mode():
        masks, iou_scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=clipped,
            multimask_output=True,
        )
    close_iterations = (
        DEFAULT_MASK_CLOSE_ITERATIONS if mask_close_iterations is None else mask_close_iterations
    )
    selected = _select_best_multimask(masks, iou_scores).astype(bool)

    stage_masks: dict[str, np.ndarray] | None = None
    if collect_stages:
        stage_masks = _postprocess_mask_stages(
            selected,
            clipped,
            close_iterations=close_iterations,
            close_kernel_size=mask_close_kernel_size,
            dilate_iterations=mask_dilate_iterations,
            dilate_kernel_size=mask_dilate_kernel_size,
        )
        postprocessed = stage_masks["postprocessed"]
        if clip_to_prompt_box:
            postprocessed = _clip_mask_to_box(postprocessed, mask_clip_box)
            stage_masks["clipped"] = postprocessed.copy()
        else:
            stage_masks["clipped"] = postprocessed.copy()
    else:
        postprocessed = _postprocess_mask(
            selected,
            clipped,
            close_iterations=close_iterations,
            close_kernel_size=mask_close_kernel_size,
            dilate_iterations=mask_dilate_iterations,
            dilate_kernel_size=mask_dilate_kernel_size,
        )
        if clip_to_prompt_box:
            postprocessed = _clip_mask_to_box(postprocessed, mask_clip_box)

    prompt_debug = None
    if collect_stages:
        prompt_debug = {
            "tight_box": tight_box,
            "sam_box": clipped,
            "mask_clip_box": mask_clip_box,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "occluder_negative_uv": occluder_negative_uv,
        }

    return postprocessed.astype(np.uint8) * 255, stage_masks, prompt_debug


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
    *,
    sam_width_ratio: float | None = None,
    sam_bottom_ratio: float | None = None,
    clip_to_prompt_box: bool = False,
    mask_close_iterations: int | None = None,
    mask_close_kernel_size: int = 1,
    mask_dilate_iterations: int = 0,
    mask_dilate_kernel_size: int = 1,
    keep_largest_component: bool = False,
    include_side_negatives: bool = False,
    mask_clip_expand_ratio: float = 1.0,
    collect_debug: bool = False,
    occlusion_aware: bool = False,
    camera=None,
    max_occluder_negative_points: int = 8,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    height, width = image_rgb.shape[:2]
    empty = np.zeros((height, width), dtype=np.uint8)
    if not prompts or predictor is None:
        if collect_debug:
            return empty, {"prompts": [], "stage_unions": {}, "per_object": []}
        return empty

    predictor.set_image(image_rgb)
    sam_union = np.zeros((height, width), dtype=bool)
    occluder_mask = np.zeros((height, width), dtype=bool)
    stage_unions: dict[str, np.ndarray] = {
        "raw": np.zeros((height, width), dtype=bool),
        "fill_holes": np.zeros((height, width), dtype=bool),
        "closing": np.zeros((height, width), dtype=bool),
        "dilated": np.zeros((height, width), dtype=bool),
        "postprocessed": np.zeros((height, width), dtype=bool),
        "clipped": np.zeros((height, width), dtype=bool),
    }
    per_object_debug: list[dict[str, object]] = []

    ordered_prompts = list(prompts)
    if occlusion_aware and camera is not None:
        from vggt_omega.waymo.utils.occlusion import prompt_depth_cam

        ordered_prompts = sorted(prompts, key=lambda prompt: prompt_depth_cam(camera, prompt))

    segment_kwargs = dict(
        sam_width_ratio=sam_width_ratio,
        sam_bottom_ratio=sam_bottom_ratio,
        clip_to_prompt_box=clip_to_prompt_box,
        mask_close_iterations=mask_close_iterations,
        mask_close_kernel_size=mask_close_kernel_size,
        mask_dilate_iterations=mask_dilate_iterations,
        mask_dilate_kernel_size=mask_dilate_kernel_size,
        keep_largest_component=keep_largest_component,
        include_side_negatives=include_side_negatives,
        mask_clip_expand_ratio=mask_clip_expand_ratio,
        collect_stages=collect_debug,
    )

    for prompt in ordered_prompts:
        occluder_negative_uv = None
        if occlusion_aware and camera is not None:
            from vggt_omega.waymo.utils.occlusion import (
                is_box_fully_occluded,
                sample_occluder_negative_points,
            )

            tight_box, clipped, _ = _resolve_sam_boxes(
                prompt,
                width,
                height,
                sam_width_ratio=sam_width_ratio,
                sam_bottom_ratio=sam_bottom_ratio,
                clip_to_prompt_box=clip_to_prompt_box,
                mask_clip_expand_ratio=mask_clip_expand_ratio,
            )
            if tight_box is None or clipped is None:
                continue
            if is_box_fully_occluded(occluder_mask, tight_box):
                continue
            track_id = getattr(prompt.box, "track_id", getattr(prompt.box, "track_uuid", id(prompt.box)))
            occluder_negative_uv = sample_occluder_negative_points(
                occluder_mask,
                clipped,
                max_points=max_occluder_negative_points,
                seed=hash(track_id) % (2**32),
            )

        object_mask, stage_masks, prompt_debug = _segment_prompt_mask(
            prompt,
            predictor,
            width,
            height,
            occluder_negative_uv=occluder_negative_uv,
            **segment_kwargs,
        )
        object_bool = object_mask.astype(bool)
        sam_union |= object_bool
        if occlusion_aware:
            occluder_mask |= object_bool
        if collect_debug and stage_masks is not None and prompt_debug is not None:
            for key in stage_unions:
                if key in stage_masks:
                    stage_unions[key] |= stage_masks[key]
            per_object_debug.append(
                {
                    "prompt": prompt,
                    "stages": stage_masks,
                    "boxes": prompt_debug,
                }
            )

    final_mask = sam_union.astype(np.uint8) * 255
    if collect_debug:
        stage_unions["final"] = sam_union
        return final_mask, {
            "prompts": prompts,
            "stage_unions": stage_unions,
            "per_object": per_object_debug,
        }
    return final_mask
