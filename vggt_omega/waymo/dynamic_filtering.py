from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from vggt_omega.av2.utils.sam2_masks import DEFAULT_SAM2_MODEL_ID, load_sam2_predictor, segment_dynamic_objects_sam2
from vggt_omega.waymo.types import WaymoBox3D, WaymoFrame
from vggt_omega.waymo.geometry import PinholeCamera
from vggt_omega.waymo.metric_alignment import scale_pinhole_camera
from vggt_omega.waymo.utils.dynamic_boxes import (
    DEFAULT_BOX_FILTER_EXPAND_RATIO,
    dynamic_boxes,
    points_inside_box,
    project_dynamic_boxes_labeled,
    waymo_sam_segment_kwargs,
)
from vggt_omega.waymo.utils.dynamic_debug import save_dynamic_filter_debug
from vggt_omega.waymo.utils.image_masks import apply_exclude_masks_to_conf, mask_to_pred_grid
from vggt_omega.waymo.utils.lidar_prompts import (
    DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    DynamicObjectSamPrompt,
    load_frame_image_rgb,
    sam_prompt_for_box,
)
from vggt_omega.waymo.utils.motion import DEFAULT_MIN_BOX_DISPLACEMENT_M, moving_track_ids
from vggt_omega.waymo.utils.scale_error import DEFAULT_SCALE_ERROR_THRESHOLD, box_depth_scale_error

__all__ = [
    "DEFAULT_BOX_FILTER_EXPAND_RATIO",
    "DEFAULT_MIN_BOX_DISPLACEMENT_M",
    "DEFAULT_SCALE_ERROR_THRESHOLD",
    "DepthPointFilter",
    "DynamicFilterMode",
    "InsideDynamicBoxFilter",
    "SelectiveInsideDynamicBoxFilter",
    "apply_dynamic_filter_to_predictions",
    "build_combined_point_filters",
]

DynamicFilterMode = Literal["combined", "sam2", "box", "none"]


class DepthPointFilter(Protocol):
    def keep(self, points_ego: np.ndarray, frame: WaymoFrame) -> np.ndarray:
        """Return a boolean mask of ego-frame points to keep."""


class InsideDynamicBoxFilter:
    def __init__(
        self,
        moving_tracks: frozenset[str] | None = None,
        *,
        box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    ):
        self._moving_tracks = moving_tracks
        self._box_expand_ratio = box_expand_ratio

    def keep(self, points_ego: np.ndarray, frame: WaymoFrame) -> np.ndarray:
        keep = np.ones(len(points_ego), dtype=bool)
        for box in dynamic_boxes(frame, moving_tracks=self._moving_tracks):
            keep &= ~points_inside_box(points_ego, box, expand_ratio=self._box_expand_ratio)
        return keep


class SelectiveInsideDynamicBoxFilter:
    def __init__(
        self,
        boxes_by_timestamp: dict[int, tuple[WaymoBox3D, ...]],
        *,
        box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    ):
        self._boxes_by_timestamp = boxes_by_timestamp
        self._box_expand_ratio = box_expand_ratio

    def keep(self, points_ego: np.ndarray, frame: WaymoFrame) -> np.ndarray:
        boxes = self._boxes_by_timestamp.get(frame.timestamp_us, ())
        if not boxes:
            return np.ones(len(points_ego), dtype=bool)
        keep = np.ones(len(points_ego), dtype=bool)
        for box in boxes:
            keep &= ~points_inside_box(points_ego, box, expand_ratio=self._box_expand_ratio)
        return keep


def build_combined_prompts(
    frame: WaymoFrame,
    native_camera: PinholeCamera,
    pred_camera: PinholeCamera,
    pred_depth: np.ndarray,
    *,
    scale_error_threshold: float,
    max_lidar_points: int,
    moving_tracks: frozenset[str],
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
) -> tuple[list[DynamicObjectSamPrompt], tuple[WaymoBox3D, ...]]:
    prompts: list[DynamicObjectSamPrompt] = []
    boxes_3d: list[WaymoBox3D] = []

    for box, xyxy in project_dynamic_boxes_labeled(
        frame, native_camera, moving_tracks=moving_tracks, expand_ratio=box_expand_ratio
    ):
        scale_error = box_depth_scale_error(box, frame, pred_camera, pred_depth)
        use_3d_box = scale_error is not None and scale_error <= scale_error_threshold
        prompts.append(
            sam_prompt_for_box(
                frame,
                box,
                xyxy,
                native_camera,
                max_lidar_points=max_lidar_points,
                scale_error=scale_error,
                use_3d_box=use_3d_box,
            )
        )
        if use_3d_box:
            boxes_3d.append(box)

    return prompts, tuple(boxes_3d)


def build_combined_point_filters(
    predictions: dict[str, np.ndarray],
    frames: list[WaymoFrame],
    native_camera: PinholeCamera,
    *,
    moving_tracks: frozenset[str],
    scale_error_threshold: float = DEFAULT_SCALE_ERROR_THRESHOLD,
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    pred_height: int | None = None,
    pred_width: int | None = None,
) -> list[DepthPointFilter] | None:
    pred_height = pred_height or predictions["depth_conf"].shape[-2]
    pred_width = pred_width or predictions["depth_conf"].shape[-1]
    pred_camera = scale_pinhole_camera(native_camera, pred_width, pred_height)

    boxes_by_timestamp: dict[int, tuple[WaymoBox3D, ...]] = {}
    for index, frame in enumerate(frames):
        boxes_3d: list[WaymoBox3D] = []
        for box, _ in project_dynamic_boxes_labeled(
            frame, native_camera, moving_tracks=moving_tracks, expand_ratio=box_expand_ratio
        ):
            scale_error = box_depth_scale_error(box, frame, pred_camera, predictions["depth"][index])
            if scale_error is not None and scale_error <= scale_error_threshold:
                boxes_3d.append(box)
        if boxes_3d:
            boxes_by_timestamp[frame.timestamp_us] = tuple(boxes_3d)

    if not boxes_by_timestamp:
        return None
    return [SelectiveInsideDynamicBoxFilter(boxes_by_timestamp, box_expand_ratio=box_expand_ratio)]


def apply_dynamic_filter_to_predictions(
    predictions: dict[str, np.ndarray],
    frames: list[WaymoFrame],
    image_paths: list[Path | str],
    native_camera: PinholeCamera,
    *,
    mode: DynamicFilterMode = "combined",
    crop_bottom: int = 0,
    scale_error_threshold: float = DEFAULT_SCALE_ERROR_THRESHOLD,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str = "cuda",
    sam2_cache_dir: str | Path | None = None,
    pred_height: int | None = None,
    pred_width: int | None = None,
    max_lidar_points: int = DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    min_box_displacement_m: float = DEFAULT_MIN_BOX_DISPLACEMENT_M,
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    debug_dir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], list[DepthPointFilter] | None]:
    if mode == "none":
        return predictions, None

    moving_tracks = moving_track_ids(frames, min_displacement_m=min_box_displacement_m)

    if mode == "box":
        return predictions, [InsideDynamicBoxFilter(moving_tracks, box_expand_ratio=box_expand_ratio)]

    pred_height = pred_height or predictions["depth_conf"].shape[-2]
    pred_width = pred_width or predictions["depth_conf"].shape[-1]
    pred_camera = scale_pinhole_camera(native_camera, pred_width, pred_height)
    predictor = load_sam2_predictor(sam2_model_id, device)

    if sam2_cache_dir is not None:
        os.makedirs(sam2_cache_dir, exist_ok=True)
    cache_dir = Path(sam2_cache_dir) if sam2_cache_dir else None

    conf = predictions["depth_conf"]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 2:
        conf = conf[np.newaxis]

    boxes_by_timestamp: dict[int, tuple[WaymoBox3D, ...]] = {}
    exclude_masks: list[np.ndarray] = []
    sam_kwargs = waymo_sam_segment_kwargs()
    debug_enabled = debug_dir is not None
    debug_prompts: list[list[DynamicObjectSamPrompt]] = []
    debug_masks: list[np.ndarray] = []
    debug_info: list[dict[str, object]] = []

    for index, (frame, image_path) in enumerate(zip(frames, image_paths, strict=True)):
        if mode == "combined":
            prompts, boxes_3d = build_combined_prompts(
                frame,
                native_camera,
                pred_camera,
                predictions["depth"][index],
                scale_error_threshold=scale_error_threshold,
                max_lidar_points=max_lidar_points,
                moving_tracks=moving_tracks,
                box_expand_ratio=box_expand_ratio,
            )
            if boxes_3d:
                boxes_by_timestamp[frame.timestamp_us] = boxes_3d
        else:
            prompts = [
                sam_prompt_for_box(
                    frame,
                    box,
                    xyxy,
                    native_camera,
                    max_lidar_points=max_lidar_points,
                )
                for box, xyxy in project_dynamic_boxes_labeled(
                    frame, native_camera, moving_tracks=moving_tracks, expand_ratio=box_expand_ratio
                )
            ]

        image_rgb = load_frame_image_rgb(image_path, crop_bottom=crop_bottom)
        cache_path = cache_dir / f"{frame.timestamp_us}.png" if cache_dir else None
        if cache_path is not None and cache_path.exists() and not debug_enabled:
            native_mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        elif debug_enabled:
            native_mask, frame_debug = segment_dynamic_objects_sam2(
                image_rgb,
                prompts,
                predictor,
                camera=native_camera,
                collect_debug=True,
                **sam_kwargs,
            )
            debug_info.append(frame_debug)
            if cache_path is not None:
                cv2.imwrite(str(cache_path), native_mask)
        else:
            native_mask = segment_dynamic_objects_sam2(
                image_rgb,
                prompts,
                predictor,
                camera=native_camera,
                **sam_kwargs,
            )
            if cache_path is not None:
                cv2.imwrite(str(cache_path), native_mask)

        if debug_enabled:
            debug_prompts.append(prompts)
            debug_masks.append(native_mask)

        exclude_masks.append(
            mask_to_pred_grid(image_rgb, native_mask, pred_height=pred_height, pred_width=pred_width)
        )

    predictions = dict(predictions)
    predictions["depth_conf"] = apply_exclude_masks_to_conf(conf, exclude_masks)

    point_filters: list[DepthPointFilter] | None = None
    if boxes_by_timestamp:
        point_filters = [SelectiveInsideDynamicBoxFilter(boxes_by_timestamp, box_expand_ratio=box_expand_ratio)]

    if debug_enabled:
        save_dynamic_filter_debug(
            frames,
            image_paths,
            native_camera,
            debug_dir,
            prompts_per_frame=debug_prompts,
            masks_per_frame=debug_masks,
            debug_info_per_frame=debug_info,
            crop_bottom=crop_bottom,
            pred_height=pred_height,
            pred_width=pred_width,
        )

    return predictions, point_filters
