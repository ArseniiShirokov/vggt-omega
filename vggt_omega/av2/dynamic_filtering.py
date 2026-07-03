from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.camera.pinhole_camera import PinholeCamera

from vggt_omega.av2.dataset import AV2Box3D, AV2Frame
from vggt_omega.av2.metric_alignment import scale_pinhole_camera
from vggt_omega.av2.utils.dynamic_boxes import (
    DEFAULT_BOX_FILTER_EXPAND_RATIO,
    dynamic_boxes,
    points_inside_box,
    project_dynamic_boxes_labeled,
)
from vggt_omega.av2.utils.dynamic_debug import save_dynamic_filter_debug
from vggt_omega.av2.utils.image_masks import apply_exclude_masks_to_conf, mask_to_pred_grid
from vggt_omega.av2.utils.lidar_prompts import (
    DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    DynamicObjectSamPrompt,
    load_frame_image_rgb,
    sam_prompt_for_box,
)
from vggt_omega.av2.utils.sam2_masks import DEFAULT_SAM2_MODEL_ID, load_sam2_predictor, segment_dynamic_objects_sam2
from vggt_omega.av2.utils.motion import DEFAULT_MIN_BOX_DISPLACEMENT_M, moving_track_uuids
from vggt_omega.av2.utils.scale_error import DEFAULT_SCALE_ERROR_THRESHOLD, box_depth_scale_error

__all__ = [
    "DEFAULT_BOX_FILTER_EXPAND_RATIO",
    "DEFAULT_MIN_BOX_DISPLACEMENT_M",
    "DEFAULT_SCALE_ERROR_THRESHOLD",
    "DepthPointFilter",
    "DynamicFilterMode",
    "InsideDynamicBoxFilter",
    "SelectiveInsideDynamicBoxFilter",
    "apply_dynamic_filter_to_predictions",
]

DynamicFilterMode = Literal["combined", "sam2", "box", "none"]


class DepthPointFilter(Protocol):
    def keep(self, points_ego: np.ndarray, frame: AV2Frame) -> np.ndarray:
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

    def keep(self, points_ego: np.ndarray, frame: AV2Frame) -> np.ndarray:
        keep = np.ones(len(points_ego), dtype=bool)
        for box in dynamic_boxes(frame, moving_tracks=self._moving_tracks):
            keep &= ~points_inside_box(
                points_ego, box, expand_ratio=self._box_expand_ratio
            )
        return keep


class SelectiveInsideDynamicBoxFilter:
    def __init__(
        self,
        boxes_by_timestamp: dict[int, tuple[AV2Box3D, ...]],
        *,
        box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    ):
        self._boxes_by_timestamp = boxes_by_timestamp
        self._box_expand_ratio = box_expand_ratio

    def keep(self, points_ego: np.ndarray, frame: AV2Frame) -> np.ndarray:
        boxes = self._boxes_by_timestamp.get(frame.cam_timestamp_ns, ())
        if not boxes:
            return np.ones(len(points_ego), dtype=bool)
        keep = np.ones(len(points_ego), dtype=bool)
        for box in boxes:
            keep &= ~points_inside_box(
                points_ego, box, expand_ratio=self._box_expand_ratio
            )
        return keep


def build_combined_prompts(
    frame: AV2Frame,
    native_camera: PinholeCamera,
    pred_camera: PinholeCamera,
    loader: AV2SensorDataLoader,
    pred_depth: np.ndarray,
    *,
    scale_error_threshold: float,
    max_lidar_points: int,
    moving_tracks: frozenset[str],
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
) -> tuple[list[DynamicObjectSamPrompt], tuple[AV2Box3D, ...]]:
    """SAM for every moving box; additionally mark low-error boxes for 3D merge filtering."""
    prompts: list[DynamicObjectSamPrompt] = []
    boxes_3d: list[AV2Box3D] = []

    for box, xyxy in project_dynamic_boxes_labeled(
        frame, native_camera, moving_tracks=moving_tracks, expand_ratio=box_expand_ratio
    ):
        scale_error = box_depth_scale_error(box, frame, loader, pred_camera, pred_depth)
        use_3d_box = scale_error is not None and scale_error <= scale_error_threshold
        prompts.append(
            sam_prompt_for_box(
                frame,
                box,
                xyxy,
                native_camera,
                loader,
                max_lidar_points=max_lidar_points,
                scale_error=scale_error,
                use_3d_box=use_3d_box,
            )
        )
        if use_3d_box:
            boxes_3d.append(box)

    return prompts, tuple(boxes_3d)


def apply_dynamic_filter_to_predictions(
    predictions: dict[str, np.ndarray],
    frames: list[AV2Frame],
    image_paths: list[Path | str],
    native_camera: PinholeCamera,
    data_root: str | Path,
    *,
    mode: DynamicFilterMode = "combined",
    crop_bottom: int = 0,
    scale_error_threshold: float = DEFAULT_SCALE_ERROR_THRESHOLD,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str = "cuda",
    sam2_cache_dir: str | Path | None = None,
    debug_dir: str | Path | None = None,
    pred_height: int | None = None,
    pred_width: int | None = None,
    max_lidar_points: int = DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    min_box_displacement_m: float = DEFAULT_MIN_BOX_DISPLACEMENT_M,
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
) -> tuple[dict[str, np.ndarray], list[DepthPointFilter] | None]:
    """Filter dynamics: SAM mask on conf for all boxes + 3D in-box filter when scale is good."""
    if mode == "none":
        return predictions, None

    moving_tracks = moving_track_uuids(frames, min_displacement_m=min_box_displacement_m)

    if mode == "box":
        if debug_dir is not None:
            loader = AV2SensorDataLoader(data_dir=Path(data_root), labels_dir=Path(data_root))
            save_dynamic_filter_debug(
                frames,
                image_paths,
                native_camera,
                debug_dir,
                prompts_per_frame=[
                    [
                        sam_prompt_for_box(frame, box, xyxy, native_camera, loader, use_3d_box=True)
                        for box, xyxy in project_dynamic_boxes_labeled(
                            frame, native_camera, moving_tracks=moving_tracks, expand_ratio=box_expand_ratio
                        )
                    ]
                    for frame in frames
                ],
                masks_per_frame=[
                    np.zeros(load_frame_image_rgb(path, crop_bottom=crop_bottom).shape[:2], dtype=np.uint8)
                    for path in image_paths
                ],
                crop_bottom=crop_bottom,
                pred_height=pred_height or predictions["depth_conf"].shape[-2],
                pred_width=pred_width or predictions["depth_conf"].shape[-1],
                load_image=load_frame_image_rgb,
            )
        return predictions, [InsideDynamicBoxFilter(moving_tracks, box_expand_ratio=box_expand_ratio)]

    pred_height = pred_height or predictions["depth_conf"].shape[-2]
    pred_width = pred_width or predictions["depth_conf"].shape[-1]
    pred_camera = scale_pinhole_camera(native_camera, pred_width, pred_height)
    loader = AV2SensorDataLoader(data_dir=Path(data_root), labels_dir=Path(data_root))
    predictor = load_sam2_predictor(sam2_model_id, device)

    if sam2_cache_dir is not None:
        os.makedirs(sam2_cache_dir, exist_ok=True)
    cache_dir = Path(sam2_cache_dir) if sam2_cache_dir else None

    conf = predictions["depth_conf"]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 2:
        conf = conf[np.newaxis]

    boxes_by_timestamp: dict[int, tuple[AV2Box3D, ...]] = {}
    exclude_masks: list[np.ndarray] = []
    prompts_per_frame: list[list[DynamicObjectSamPrompt]] = []
    native_masks: list[np.ndarray] = []

    for index, (frame, image_path) in enumerate(zip(frames, image_paths, strict=True)):
        if mode == "combined":
            prompts, boxes_3d = build_combined_prompts(
                frame,
                native_camera,
                pred_camera,
                loader,
                predictions["depth"][index],
                scale_error_threshold=scale_error_threshold,
                max_lidar_points=max_lidar_points,
                moving_tracks=moving_tracks,
                box_expand_ratio=box_expand_ratio,
            )
            if boxes_3d:
                boxes_by_timestamp[frame.cam_timestamp_ns] = boxes_3d
        else:
            prompts = [
                sam_prompt_for_box(frame, box, xyxy, native_camera, loader, max_lidar_points=max_lidar_points)
                for box, xyxy in project_dynamic_boxes_labeled(
                    frame, native_camera, moving_tracks=moving_tracks, expand_ratio=box_expand_ratio
                )
            ]

        prompts_per_frame.append(prompts)
        image_rgb = load_frame_image_rgb(image_path, crop_bottom=crop_bottom)

        cache_path = cache_dir / f"{frame.cam_timestamp_ns}.png" if cache_dir else None
        if cache_path is not None and cache_path.exists():
            native_mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        else:
            native_mask = segment_dynamic_objects_sam2(image_rgb, prompts, predictor)
            if cache_path is not None:
                cv2.imwrite(str(cache_path), native_mask)

        native_masks.append(native_mask)
        exclude_masks.append(
            mask_to_pred_grid(image_rgb, native_mask, pred_height=pred_height, pred_width=pred_width)
        )

    if debug_dir is not None:
        save_dynamic_filter_debug(
            frames,
            image_paths,
            native_camera,
            debug_dir,
            prompts_per_frame=prompts_per_frame,
            masks_per_frame=native_masks,
            crop_bottom=crop_bottom,
            pred_height=pred_height,
            pred_width=pred_width,
            load_image=load_frame_image_rgb,
        )

    predictions = dict(predictions)
    predictions["depth_conf"] = apply_exclude_masks_to_conf(conf, exclude_masks)

    point_filters: list[DepthPointFilter] | None = None
    if boxes_by_timestamp:
        point_filters = [SelectiveInsideDynamicBoxFilter(boxes_by_timestamp, box_expand_ratio=box_expand_ratio)]

    return predictions, point_filters
