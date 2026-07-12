from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from vggt_omega.av2.utils.sam2_masks import DEFAULT_SAM2_MODEL_ID, load_sam2_predictor, segment_dynamic_objects_sam2
from vggt_omega.waymo.dataset import WaymoSceneDataset
from vggt_omega.waymo.types import WaymoFrame
from vggt_omega.waymo.inference import crop_waymo_images
from vggt_omega.waymo.metric_alignment import build_pinhole_camera
from vggt_omega.waymo.utils.dynamic_boxes import (
    DEFAULT_BOX_FILTER_EXPAND_RATIO,
    project_dynamic_boxes_labeled,
    waymo_sam_segment_kwargs,
)
from vggt_omega.waymo.utils.dynamic_debug import save_dynamic_filter_debug
from vggt_omega.waymo.utils.image_masks import apply_exclude_masks_to_conf, mask_to_pred_grid
from vggt_omega.waymo.utils.lidar_prompts import (
    DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    load_frame_image_rgb,
    sam_prompt_for_box,
)
from vggt_omega.waymo.utils.motion import moving_track_ids
from vggt_omega.utils.sky_mask import _load_skyseg_session, segment_sky


def resolve_dynamic_mask_cache_dir(
    dynamic_mask_cache_dir: str | Path | None,
    data_root: Path,
    split: str,
    scene_id: str,
) -> Path:
    if dynamic_mask_cache_dir is None:
        return data_root / split / "cache" / scene_id / "dynamic_masks"
    return Path(dynamic_mask_cache_dir) / scene_id


@dataclass(frozen=True)
class SceneMaskCache:
    sky_mask_cache_dir: Path
    dynamic_mask_cache_dir: Path
    moving_tracks: frozenset[str]


def precompute_scene_masks(
    data_root: str | Path,
    scene_id: str,
    scene: WaymoSceneDataset,
    indices: list[int],
    image_paths: list[Path],
    *,
    split: str = "training",
    crop_bottom: int = 0,
    crop_cache_dir: str | Path | None = None,
    sky_mask_cache_dir: str | Path | None = None,
    dynamic_mask_cache_dir: str | Path | None = None,
    min_box_displacement_m: float = 0.2,
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    max_lidar_points: int = DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str = "cuda",
    skyseg_model_path: str = "skyseg.onnx",
    filter_dynamic: bool = True,
    debug_dir: str | Path | None = None,
    pred_height: int | None = None,
    pred_width: int | None = None,
) -> SceneMaskCache:
    data_root = Path(data_root)
    if crop_cache_dir is None:
        crop_cache_dir = data_root / split / "cache" / scene_id / "crop"
    if sky_mask_cache_dir is None:
        sky_mask_cache_dir = data_root / split / "cache" / scene_id / "sky_masks"
    sky_dir = Path(sky_mask_cache_dir) / scene_id
    dynamic_dir = resolve_dynamic_mask_cache_dir(dynamic_mask_cache_dir, data_root, split, scene_id)
    sky_dir.mkdir(parents=True, exist_ok=True)
    dynamic_dir.mkdir(parents=True, exist_ok=True)

    inference_paths = crop_waymo_images(image_paths, crop_bottom, Path(crop_cache_dir))
    sky_session = _load_skyseg_session(skyseg_model_path)

    for inference_path in tqdm(inference_paths, desc="Sky masks", leave=False):
        mask_path = sky_dir / Path(inference_path).name
        if not mask_path.exists():
            segment_sky(inference_path, sky_session, str(mask_path))

    moving_tracks: frozenset[str] = frozenset()
    if filter_dynamic:
        metadata_frames = [scene.get_frame(index, load_lidar=False) for index in indices]
        moving_tracks = moving_track_ids(metadata_frames, min_displacement_m=min_box_displacement_m)
        native_camera = build_pinhole_camera(metadata_frames[0], crop_bottom)
        predictor = load_sam2_predictor(sam2_model_id, device)
        sam_kwargs = waymo_sam_segment_kwargs()
        debug_enabled = debug_dir is not None
        if debug_enabled:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

        debug_frames: list[WaymoFrame] = []
        debug_image_paths: list[Path] = []
        debug_prompts: list[list] = []
        debug_masks: list[np.ndarray] = []
        debug_info: list[dict[str, object]] = []

        for index, image_path in tqdm(
            zip(indices, image_paths, strict=True),
            total=len(indices),
            desc="Dynamic masks",
            leave=False,
        ):
            frame_meta = scene.get_frame(index, load_lidar=False)
            cache_path = dynamic_dir / f"{frame_meta.timestamp_us}.png"
            recompute = debug_enabled or not cache_path.exists()

            frame = scene.get_frame(index, load_lidar=True, use_side_lidar=False)
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

            if recompute:
                if debug_enabled:
                    native_mask, frame_debug = segment_dynamic_objects_sam2(
                        image_rgb,
                        prompts,
                        predictor,
                        camera=native_camera,
                        collect_debug=True,
                        **sam_kwargs,
                    )
                    debug_info.append(frame_debug)
                else:
                    native_mask = segment_dynamic_objects_sam2(
                        image_rgb,
                        prompts,
                        predictor,
                        camera=native_camera,
                        **sam_kwargs,
                    )
                cv2.imwrite(str(cache_path), native_mask)
            else:
                native_mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
                if debug_enabled:
                    _, frame_debug = segment_dynamic_objects_sam2(
                        image_rgb,
                        prompts,
                        predictor,
                        camera=native_camera,
                        collect_debug=True,
                        **sam_kwargs,
                    )
                    debug_info.append(frame_debug)

            if debug_enabled:
                debug_frames.append(frame)
                debug_image_paths.append(image_path)
                debug_prompts.append(prompts)
                debug_masks.append(native_mask)

        if debug_enabled:
            save_dynamic_filter_debug(
                debug_frames,
                debug_image_paths,
                native_camera,
                debug_dir,
                prompts_per_frame=debug_prompts,
                masks_per_frame=debug_masks,
                debug_info_per_frame=debug_info,
                crop_bottom=crop_bottom,
                pred_height=pred_height,
                pred_width=pred_width,
            )

    return SceneMaskCache(
        sky_mask_cache_dir=sky_dir,
        dynamic_mask_cache_dir=dynamic_dir,
        moving_tracks=moving_tracks,
    )


def _load_sky_mask_on_pred_grid(
    inference_path: str | Path,
    image_path: Path | str,
    sky_mask_cache_dir: Path,
    *,
    crop_bottom: int,
    pred_height: int,
    pred_width: int,
) -> np.ndarray:
    mask_path = sky_mask_cache_dir / Path(inference_path).name
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing precomputed sky mask: {mask_path}")

    sky_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image_rgb = load_frame_image_rgb(image_path, crop_bottom=crop_bottom)
    return mask_to_pred_grid(image_rgb, sky_mask, pred_height=pred_height, pred_width=pred_width)


def _load_dynamic_mask_on_pred_grid(
    frame: WaymoFrame,
    image_path: Path | str,
    dynamic_mask_cache_dir: Path,
    *,
    crop_bottom: int,
    pred_height: int,
    pred_width: int,
) -> np.ndarray:
    mask_path = dynamic_mask_cache_dir / f"{frame.timestamp_us}.png"
    if not mask_path.is_file():
        return np.zeros((pred_height, pred_width), dtype=np.float32)

    dynamic_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image_rgb = load_frame_image_rgb(image_path, crop_bottom=crop_bottom)
    return mask_to_pred_grid(image_rgb, dynamic_mask, pred_height=pred_height, pred_width=pred_width)


def apply_scene_masks_to_predictions(
    predictions: dict[str, np.ndarray],
    frames: list[WaymoFrame],
    image_paths: list[Path | str],
    inference_paths: list[str | Path],
    mask_cache: SceneMaskCache,
    *,
    crop_bottom: int = 0,
    apply_sky: bool = True,
    apply_dynamic: bool = True,
) -> dict[str, np.ndarray]:
    conf = predictions["depth_conf"]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 2:
        conf = conf[np.newaxis]

    pred_height, pred_width = conf.shape[-2], conf.shape[-1]
    exclude_masks: list[np.ndarray] = []

    for frame, image_path, inference_path in zip(frames, image_paths, inference_paths, strict=True):
        exclude = np.zeros((pred_height, pred_width), dtype=np.float32)

        if apply_sky:
            sky_on_grid = _load_sky_mask_on_pred_grid(
                inference_path,
                image_path,
                mask_cache.sky_mask_cache_dir,
                crop_bottom=crop_bottom,
                pred_height=pred_height,
                pred_width=pred_width,
            )
            exclude = np.maximum(exclude, (sky_on_grid <= 0.1).astype(np.float32))

        if apply_dynamic:
            dynamic_on_grid = _load_dynamic_mask_on_pred_grid(
                frame,
                image_path,
                mask_cache.dynamic_mask_cache_dir,
                crop_bottom=crop_bottom,
                pred_height=pred_height,
                pred_width=pred_width,
            )
            exclude = np.maximum(exclude, (dynamic_on_grid > 0.1).astype(np.float32))

        exclude_masks.append(exclude)

    predictions = dict(predictions)
    predictions["depth_conf"] = apply_exclude_masks_to_conf(conf, exclude_masks)
    return predictions


def resolve_scene_frame_range(
    scene: WaymoSceneDataset,
    frame_start: int | None,
    frame_end: int | None,
) -> list[int]:
    start = 0 if frame_start is None else frame_start
    end = (len(scene) - 1) if frame_end is None else frame_end
    if start < 0 or end >= len(scene) or end < start:
        raise IndexError(
            f"Invalid frame range [{start}, {end}] for scene with {len(scene)} frames"
        )
    usable = scene.usable_indices(start, end)
    if not usable:
        raise ValueError(f"No usable frames (LiDAR) in range [{start}, {end}]")
    return usable
