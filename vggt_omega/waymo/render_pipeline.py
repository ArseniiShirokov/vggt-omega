from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from vggt_omega.rendering.zbuffer import render_points_in_camera, splat_points_in_camera
from vggt_omega.waymo.dataset import DEFAULT_WAYMO_CROP_BOTTOM, WaymoSceneDataset
from vggt_omega.waymo.types import WaymoFrame
from vggt_omega.waymo.dynamic_filtering import (
    DEFAULT_BOX_FILTER_EXPAND_RATIO,
    DEFAULT_MIN_BOX_DISPLACEMENT_M,
    DEFAULT_SCALE_ERROR_THRESHOLD,
    DepthPointFilter,
    InsideDynamicBoxFilter,
    apply_dynamic_filter_to_predictions,
    build_combined_point_filters,
)
from vggt_omega.waymo.inference import run_vggt_on_waymo_chunk
from vggt_omega.waymo.metric_alignment import (
    build_pinhole_camera,
    depth_to_cam_points,
    motion_compensate_ego,
    scale_pinhole_camera,
)
from vggt_omega.waymo.pipeline_config import (
    DEFAULT_RENDER_PIPELINE_CONFIG,
    config_defaults,
    load_render_pipeline_config,
)
from vggt_omega.waymo.scene_masks import (
    SceneMaskCache,
    apply_scene_masks_to_predictions,
    precompute_scene_masks,
    resolve_dynamic_mask_cache_dir,
    resolve_scene_frame_range,
)
from vggt_omega.waymo.utils.lidar_prompts import DEFAULT_MAX_LIDAR_PROMPT_POINTS
from vggt_omega.waymo.utils.motion import moving_track_ids
from vggt_omega.av2.utils.sam2_masks import DEFAULT_SAM2_MODEL_ID


@dataclass
class MergedPointCloud:
    """Fused point cloud in the ego frame of the first camera timestamp."""

    points_ego0: np.ndarray
    points_cam0: np.ndarray
    colors: np.ndarray


def sample_image_colors(
    image_path: Path,
    uv: np.ndarray,
    *,
    camera_width: int,
    camera_height: int,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    u = np.clip(np.round(uv[:, 0] * width / camera_width).astype(np.int32), 0, width - 1)
    v = np.clip(np.round(uv[:, 1] * height / camera_height).astype(np.int32), 0, height - 1)
    return image[v, u]


def merge_depth_maps_in_ego0(
    predictions: dict[str, np.ndarray],
    frames: list[WaymoFrame],
    camera,
    color_image_paths: list[Path],
    *,
    conf_percentile: float = 0.0,
    point_filters: list[DepthPointFilter] | None = None,
) -> MergedPointCloud:
    """Fuse metric depth maps into the ego frame at the first camera timestamp."""
    frame_0 = frames[0]
    cam_SE3_ego = camera.ego_SE3_cam.inverse()

    all_points_ego0: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for index, frame in enumerate(frames):
        depth = predictions["depth"][index]
        conf = predictions["depth_conf"][index].squeeze()
        points_cam, uv = depth_to_cam_points(depth, camera)

        depth_h, depth_w = depth.shape[:2]
        conf_u = np.clip(np.round(uv[:, 0] * depth_w / camera.width_px).astype(np.int32), 0, depth_w - 1)
        conf_v = np.clip(np.round(uv[:, 1] * depth_h / camera.height_px).astype(np.int32), 0, depth_h - 1)
        pixel_conf = conf[conf_v, conf_u]
        keep = pixel_conf > 1e-5
        if conf_percentile > 0:
            valid_conf = conf[np.isfinite(conf) & (conf > 1e-5)]
            threshold = np.percentile(valid_conf, conf_percentile)
            keep &= pixel_conf >= threshold

        points_cam = points_cam[keep]
        uv = uv[keep]
        points_ego = camera.ego_SE3_cam.transform_point_cloud(points_cam)
        if point_filters:
            keep_ego = np.ones(len(points_ego), dtype=bool)
            for point_filter in point_filters:
                keep_ego &= point_filter.keep(points_ego, frame)
            points_ego = points_ego[keep_ego]
            uv = uv[keep_ego]
        points_ego0 = motion_compensate_ego(points_ego, frame.world_SE3_ego, frame_0.world_SE3_ego)

        all_points_ego0.append(points_ego0)
        all_colors.append(
            sample_image_colors(
                color_image_paths[index],
                uv,
                camera_width=camera.width_px,
                camera_height=camera.height_px,
            )
        )

    points_ego0 = np.concatenate(all_points_ego0, axis=0)
    points_cam0 = cam_SE3_ego.transform_point_cloud(points_ego0)

    return MergedPointCloud(
        points_ego0=points_ego0,
        points_cam0=points_cam0,
        colors=np.concatenate(all_colors, axis=0),
    )


def save_rendered_depth_npz(depth: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, depth=depth.astype(np.float32))
    return output_path


def resolve_scene_output_dir(output_dir: str | Path, scene_id: str) -> Path:
    return Path(output_dir) / "scenes" / scene_id


def load_gt_image(
    image_path: Path,
    *,
    crop_bottom: int,
    width: int,
    height: int,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_bottom > 0:
        image = image[: image.shape[0] - crop_bottom]
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def stitch_gt_render_comparison(
    gt_rgb: np.ndarray,
    render_rgb: np.ndarray,
    *,
    gt_label: str = "GT",
    render_label: str = "Render",
) -> np.ndarray:
    divider = np.full((gt_rgb.shape[0], 2, 3), 64, dtype=np.uint8)
    combined = np.concatenate([gt_rgb, divider, render_rgb], axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, x_offset in ((gt_label, 12), (render_label, gt_rgb.shape[1] + divider.shape[1] + 12)):
        cv2.putText(combined, label, (x_offset, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, label, (x_offset, 28), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return combined


def save_comparison_gif(
    gt_image_paths: list[Path],
    rendered_paths: list[Path],
    output_path: Path,
    *,
    crop_bottom: int,
    width: int,
    height: int,
    fps: float = 10.0,
) -> Path:
    if len(gt_image_paths) != len(rendered_paths):
        raise ValueError("GT and render frame counts must match")

    frames: list[Image.Image] = []
    duration_ms = max(1, int(round(1000.0 / fps)))
    for gt_path, render_path in zip(gt_image_paths, rendered_paths, strict=True):
        gt_rgb = load_gt_image(gt_path, crop_bottom=crop_bottom, width=width, height=height)
        render_bgr = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        if render_bgr is None:
            raise FileNotFoundError(f"Failed to read render image: {render_path}")
        render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(stitch_gt_render_comparison(gt_rgb, render_rgb)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def render_all_camera_views(
    merged: MergedPointCloud,
    frames: list[WaymoFrame],
    camera,
    output_dir: Path,
) -> list[Path]:
    frame_0 = frames[0]
    cam_SE3_ego = camera.ego_SE3_cam.inverse()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []

    for index, frame in enumerate(frames):
        points_ego = motion_compensate_ego(merged.points_ego0, frame_0.world_SE3_ego, frame.world_SE3_ego)
        points_cam = cam_SE3_ego.transform_point_cloud(points_ego)
        rendered = render_points_in_camera(camera, points_cam, merged.colors)
        output_path = output_dir / f"{index:04d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        rendered_paths.append(output_path)

    return rendered_paths


def render_merged_at_frame(
    merged: MergedPointCloud,
    merge_frame_0: WaymoFrame,
    render_frame: WaymoFrame,
    camera,
) -> tuple[np.ndarray, np.ndarray]:
    cam_SE3_ego = camera.ego_SE3_cam.inverse()
    points_ego = motion_compensate_ego(
        merged.points_ego0,
        merge_frame_0.world_SE3_ego,
        render_frame.world_SE3_ego,
    )
    points_cam = cam_SE3_ego.transform_point_cloud(points_ego)
    return splat_points_in_camera(camera, points_cam, merged.colors)


def save_render_video(
    rendered_paths: list[Path],
    output_path: Path,
    *,
    fps: float = 10.0,
) -> Path:
    if not rendered_paths:
        raise ValueError("No renders to save into video")

    first = cv2.imread(str(rendered_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read render image: {rendered_paths[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for path in rendered_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Failed to read render image: {path}")
            if frame.shape[:2] != first.shape[:2]:
                frame = cv2.resize(frame, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def run_sliding_window_scene_pipeline(
    data_root: str | Path,
    scene_id: str,
    frame_start: int | None,
    frame_end: int | None,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "training",
    merge_frames: int = 8,
    target_fps: float = 10.0,
    image_resolution: int = 512,
    preprocess_mode: str = "max_size",
    device: str = "cuda",
    crop_bottom_pixels: int = DEFAULT_WAYMO_CROP_BOTTOM,
    crop_cache_dir: str | Path | None = None,
    sky_mask_cache_dir: str | Path | None = None,
    image_cache_dir: str | Path | None = None,
    conf_percentile: float = 0.0,
    align_metric: bool = True,
    save_comparison: bool = True,
    comparison_fps: float | None = None,
    filter_dynamic: bool = True,
    dynamic_filter_mode: str = "combined",
    scale_error_threshold: float = DEFAULT_SCALE_ERROR_THRESHOLD,
    min_box_displacement_m: float = DEFAULT_MIN_BOX_DISPLACEMENT_M,
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    max_lidar_points: int = DEFAULT_MAX_LIDAR_PROMPT_POINTS,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    dynamic_mask_cache_dir: str | Path | None = None,
    skip_mask_precompute: bool = False,
    debug_dynamic_filter: bool = False,
) -> tuple[Path, list[Path], list[Path], Path | None, Path | None]:
    output_dir = Path(output_dir)
    data_root = Path(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = WaymoSceneDataset(
        data_root,
        scene_id,
        split=split,
        target_fps=target_fps,
        image_cache_dir=image_cache_dir,
    )
    usable_indices = resolve_scene_frame_range(scene, frame_start, frame_end)
    scene_image_paths = [scene.image_path_at(index) for index in usable_indices]

    if len(usable_indices) < merge_frames + 1:
        raise ValueError(
            f"Only {len(usable_indices)} usable frames; need at least {merge_frames + 1} for sliding window"
        )

    scene_dir = resolve_scene_output_dir(output_dir, scene_id)
    debug_dir = scene_dir / "dynamic_debug" if debug_dynamic_filter else None

    mask_cache: SceneMaskCache | None = None
    if not skip_mask_precompute:
        mask_cache = precompute_scene_masks(
            data_root,
            scene_id,
            scene,
            usable_indices,
            scene_image_paths,
            split=split,
            crop_bottom=crop_bottom_pixels,
            crop_cache_dir=crop_cache_dir,
            sky_mask_cache_dir=sky_mask_cache_dir,
            dynamic_mask_cache_dir=dynamic_mask_cache_dir,
            min_box_displacement_m=min_box_displacement_m,
            box_expand_ratio=box_expand_ratio,
            max_lidar_points=max_lidar_points,
            sam2_model_id=sam2_model_id,
            device=device,
            filter_dynamic=filter_dynamic and dynamic_filter_mode in ("combined", "sam2"),
            debug_dir=debug_dir,
            pred_height=image_resolution,
            pred_width=image_resolution,
        )
    elif filter_dynamic and dynamic_filter_mode in ("combined", "sam2"):
        if sky_mask_cache_dir is None:
            sky_mask_cache_dir = data_root / split / "cache" / scene_id / "sky_masks"
        metadata_frames = [scene.get_frame(index, load_lidar=False) for index in usable_indices]
        mask_cache = SceneMaskCache(
            sky_mask_cache_dir=Path(sky_mask_cache_dir) / scene_id,
            dynamic_mask_cache_dir=resolve_dynamic_mask_cache_dir(
                dynamic_mask_cache_dir, data_root, split, scene_id
            ),
            moving_tracks=moving_track_ids(metadata_frames, min_displacement_m=min_box_displacement_m),
        )

    rendered_paths: list[Path] = []
    depth_paths: list[Path] = []
    render_image_paths: list[Path] = []
    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    window_idx = 0
    camera = None
    for pos in tqdm(range(len(usable_indices) - merge_frames), desc="Sliding window"):
        merge_indices = usable_indices[pos : pos + merge_frames]
        if merge_indices[-1] - merge_indices[0] != merge_frames - 1:
            continue

        render_dataset_index = usable_indices[pos + merge_frames]
        merge_start, merge_end = merge_indices[0], merge_indices[-1]

        predictions, image_paths, inference_paths = run_vggt_on_waymo_chunk(
            data_root=data_root,
            scene_id=scene_id,
            frame_start=merge_start,
            frame_end=merge_end,
            checkpoint_path=checkpoint_path,
            split=split,
            target_fps=target_fps,
            image_resolution=image_resolution,
            preprocess_mode=preprocess_mode,
            device=device,
            apply_sky_mask=False,
            crop_bottom_pixels=crop_bottom_pixels,
            crop_cache_dir=crop_cache_dir,
            align_metric=align_metric,
            image_cache_dir=image_cache_dir,
        )

        merge_frames_list = [scene[index] for index in merge_indices]
        render_frame = scene[render_dataset_index]
        pred_height, pred_width = predictions["depth"].shape[1:3]
        camera = scale_pinhole_camera(
            build_pinhole_camera(merge_frames_list[0], crop_bottom_pixels),
            pred_width,
            pred_height,
        )

        if mask_cache is not None:
            predictions = apply_scene_masks_to_predictions(
                predictions,
                merge_frames_list,
                image_paths,
                inference_paths,
                mask_cache,
                crop_bottom=crop_bottom_pixels,
                apply_sky=True,
                apply_dynamic=filter_dynamic and dynamic_filter_mode in ("combined", "sam2"),
            )

        point_filters: list[DepthPointFilter] | None = None
        if filter_dynamic:
            if dynamic_filter_mode == "combined" and mask_cache is not None:
                native_camera = build_pinhole_camera(merge_frames_list[0], crop_bottom_pixels)
                point_filters = build_combined_point_filters(
                    predictions,
                    merge_frames_list,
                    native_camera,
                    moving_tracks=mask_cache.moving_tracks,
                    scale_error_threshold=scale_error_threshold,
                    box_expand_ratio=box_expand_ratio,
                    pred_height=pred_height,
                    pred_width=pred_width,
                )
            elif dynamic_filter_mode == "box" and mask_cache is not None:
                point_filters = [
                    InsideDynamicBoxFilter(mask_cache.moving_tracks, box_expand_ratio=box_expand_ratio)
                ]

        merged = merge_depth_maps_in_ego0(
            predictions,
            merge_frames_list,
            camera,
            [Path(path) for path in inference_paths],
            conf_percentile=conf_percentile,
            point_filters=point_filters,
        )

        rendered_rgb, rendered_depth = render_merged_at_frame(
            merged, merge_frames_list[0], render_frame, camera
        )
        rgb_path = rgb_dir / f"{window_idx:04d}.jpg"
        depth_path = depth_dir / f"{window_idx:04d}.npz"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
        save_rendered_depth_npz(rendered_depth, depth_path)
        rendered_paths.append(rgb_path)
        depth_paths.append(depth_path)
        render_image_paths.append(render_frame.image_path)
        window_idx += 1

    video_fps = target_fps if comparison_fps is None else comparison_fps
    video_path = save_render_video(rendered_paths, scene_dir / "scene_render.mp4", fps=video_fps)

    comparison_path: Path | None = None
    if save_comparison and camera is not None:
        comparison_path = save_comparison_gif(
            render_image_paths,
            rendered_paths,
            scene_dir / "comparison.gif",
            crop_bottom=crop_bottom_pixels,
            width=camera.width_px,
            height=camera.height_px,
            fps=video_fps,
        )

    return scene_dir, rendered_paths, depth_paths, video_path, comparison_path


def run_render_pipeline(
    data_root: str | Path,
    scene_id: str,
    frame_start: int,
    frame_end: int,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "training",
    target_fps: float = 10.0,
    image_resolution: int = 512,
    preprocess_mode: str = "max_size",
    device: str = "cuda",
    crop_bottom_pixels: int = DEFAULT_WAYMO_CROP_BOTTOM,
    crop_cache_dir: str | Path | None = None,
    sky_mask_cache_dir: str | Path | None = None,
    image_cache_dir: str | Path | None = None,
    conf_percentile: float = 0.0,
    align_metric: bool = True,
    save_comparison: bool = True,
    comparison_fps: float | None = None,
    filter_dynamic: bool = True,
    dynamic_filter_mode: str = "combined",
    scale_error_threshold: float = DEFAULT_SCALE_ERROR_THRESHOLD,
    min_box_displacement_m: float = DEFAULT_MIN_BOX_DISPLACEMENT_M,
    box_expand_ratio: float = DEFAULT_BOX_FILTER_EXPAND_RATIO,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    sam2_cache_dir: str | Path | None = None,
    debug_dynamic_filter: bool = False,
) -> tuple[MergedPointCloud, list[Path], Path | None]:
    output_dir = Path(output_dir)
    data_root = Path(data_root)

    predictions, image_paths, inference_paths = run_vggt_on_waymo_chunk(
        data_root=data_root,
        scene_id=scene_id,
        frame_start=frame_start,
        frame_end=frame_end,
        checkpoint_path=checkpoint_path,
        split=split,
        target_fps=target_fps,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
        device=device,
        apply_sky_mask=True,
        sky_mask_cache_dir=sky_mask_cache_dir,
        crop_bottom_pixels=crop_bottom_pixels,
        crop_cache_dir=crop_cache_dir,
        align_metric=align_metric,
        image_cache_dir=image_cache_dir,
    )

    scene = WaymoSceneDataset(
        data_root,
        scene_id,
        split=split,
        target_fps=target_fps,
        image_cache_dir=image_cache_dir,
    )
    frames = [scene[index] for index in range(frame_start, frame_end + 1)]
    pred_height, pred_width = predictions["depth"].shape[1:3]
    camera = scale_pinhole_camera(
        build_pinhole_camera(frames[0], crop_bottom_pixels),
        pred_width,
        pred_height,
    )
    native_camera = build_pinhole_camera(frames[0], crop_bottom_pixels)

    dynamic_mode = "none" if not filter_dynamic else dynamic_filter_mode
    debug_dir = (output_dir / scene_id / "dynamic_debug") if debug_dynamic_filter else None
    predictions, point_filters = apply_dynamic_filter_to_predictions(
        predictions,
        frames,
        image_paths,
        native_camera,
        mode=dynamic_mode,
        crop_bottom=crop_bottom_pixels,
        sam2_model_id=sam2_model_id,
        device=device,
        sam2_cache_dir=sam2_cache_dir,
        pred_height=pred_height,
        pred_width=pred_width,
        scale_error_threshold=scale_error_threshold,
        min_box_displacement_m=min_box_displacement_m,
        box_expand_ratio=box_expand_ratio,
        debug_dir=debug_dir,
    )

    merged = merge_depth_maps_in_ego0(
        predictions,
        frames,
        camera,
        [Path(path) for path in inference_paths],
        conf_percentile=conf_percentile,
        point_filters=point_filters,
    )
    rendered_paths = render_all_camera_views(merged, frames, camera, output_dir / scene_id)

    np.savez(
        output_dir / scene_id / "merged_pointcloud.npz",
        points_ego0=merged.points_ego0,
        points_cam0=merged.points_cam0,
        colors=merged.colors,
    )
    np.savez(output_dir / scene_id / "predictions.npz", **predictions)

    comparison_path: Path | None = None
    if save_comparison:
        comparison_path = save_comparison_gif(
            [Path(path) for path in image_paths],
            rendered_paths,
            output_dir / scene_id / "comparison.gif",
            crop_bottom=crop_bottom_pixels,
            width=camera.width_px,
            height=camera.height_px,
            fps=target_fps if comparison_fps is None else comparison_fps,
        )

    return merged, rendered_paths, comparison_path


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RENDER_PIPELINE_CONFIG,
        help=f"YAML config with standard params (default: {DEFAULT_RENDER_PIPELINE_CONFIG})",
    )
    pre_args, remaining = pre_parser.parse_known_args()

    defaults: dict = {}
    if pre_args.config is not None:
        if not pre_args.config.is_file():
            raise FileNotFoundError(f"Pipeline config not found: {pre_args.config}")
        defaults = {
            key: value
            for key, value in config_defaults(load_render_pipeline_config(pre_args.config)).items()
            if value is not None
        }

    parser = argparse.ArgumentParser(
        description="VGGT Waymo inference, fusion, and render pipeline",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=defaults.get(
            "data_root",
            Path("/home/jovyan/datasets/self-driving/waymo/waymo_open_dataset_v_2_0_1"),
        ),
    )
    parser.add_argument("--split", default=defaults.get("split", "training"))
    parser.add_argument("--scene-id", default=defaults.get("scene_id"))
    parser.add_argument("--frame-start", type=int, default=defaults.get("frame_start"))
    parser.add_argument("--frame-end", type=int, default=defaults.get("frame_end"))
    parser.add_argument("--checkpoint", type=Path, default=defaults.get("checkpoint"))
    parser.add_argument("--output-dir", type=Path, default=defaults.get("output_dir"))
    parser.add_argument("--image-resolution", type=int, default=defaults.get("image_resolution", 512))
    parser.add_argument("--target-fps", type=float, default=defaults.get("target_fps", 10.0))
    parser.add_argument("--device", default=defaults.get("device", "cuda"))
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=defaults.get("crop_bottom", DEFAULT_WAYMO_CROP_BOTTOM),
    )
    parser.add_argument("--crop-cache-dir", type=Path, default=defaults.get("crop_cache_dir"))
    parser.add_argument("--sky-mask-cache-dir", type=Path, default=defaults.get("sky_mask_cache_dir"))
    parser.add_argument("--image-cache-dir", type=Path, default=defaults.get("image_cache_dir"))
    parser.add_argument("--conf-percentile", type=float, default=defaults.get("conf_percentile", 0.0))
    parser.add_argument(
        "--no-metric-alignment",
        action="store_true",
        default=defaults.get("no_metric_alignment", False),
    )
    parser.add_argument("--no-crop", action="store_true", default=defaults.get("no_crop", False))
    parser.add_argument(
        "--no-comparison-gif",
        action="store_true",
        default=defaults.get("no_comparison_gif", False),
    )
    parser.add_argument("--comparison-fps", type=float, default=defaults.get("comparison_fps"))
    parser.add_argument(
        "--no-dynamic-filter",
        action="store_true",
        default=defaults.get("no_dynamic_filter", False),
    )
    parser.add_argument(
        "--dynamic-filter-mode",
        choices=("combined", "sam2", "box"),
        default=defaults.get("dynamic_filter_mode", "combined"),
    )
    parser.add_argument(
        "--scale-error-threshold",
        type=float,
        default=defaults.get("scale_error_threshold", DEFAULT_SCALE_ERROR_THRESHOLD),
    )
    parser.add_argument(
        "--min-box-displacement-m",
        type=float,
        default=defaults.get("min_box_displacement_m", DEFAULT_MIN_BOX_DISPLACEMENT_M),
    )
    parser.add_argument(
        "--box-filter-expand-ratio",
        type=float,
        default=defaults.get("box_filter_expand_ratio", DEFAULT_BOX_FILTER_EXPAND_RATIO),
    )
    parser.add_argument(
        "--max-lidar-prompt-points",
        type=int,
        default=defaults.get("max_lidar_prompt_points", DEFAULT_MAX_LIDAR_PROMPT_POINTS),
    )
    parser.add_argument(
        "--sam2-model-id",
        default=defaults.get("sam2_model_id", DEFAULT_SAM2_MODEL_ID),
    )
    parser.add_argument("--sam2-cache-dir", type=Path, default=defaults.get("sam2_cache_dir"))
    parser.add_argument(
        "--dynamic-mask-cache-dir",
        type=Path,
        default=defaults.get("dynamic_mask_cache_dir"),
    )
    parser.add_argument(
        "--sliding-window",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("sliding_window", False),
    )
    parser.add_argument(
        "--merge-frames",
        type=int,
        default=defaults.get("merge_frames", 8),
    )
    parser.add_argument(
        "--skip-mask-precompute",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("skip_mask_precompute", False),
    )
    parser.add_argument(
        "--debug-dynamic-filter",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("debug_dynamic_filter", False),
        help="Save SAM prompts, per-stage masks, and overlays under scene dynamic_debug/",
    )

    parser.parse_args(remaining, namespace=pre_args)
    args = pre_args
    if args.scene_id is not None:
        args.scene_id = str(args.scene_id)

    if args.sliding_window:
        required = ("scene_id", "checkpoint", "output_dir")
    else:
        required = ("scene_id", "frame_start", "frame_end", "checkpoint", "output_dir")
    missing = [name for name in required if getattr(args, name.replace("-", "_"), None) is None]
    if missing:
        parser.error(
            "Missing required arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            + f" (set them in {args.config} or pass on CLI)"
        )
    return args


def main() -> None:
    args = parse_args()
    crop_bottom = 0 if args.no_crop else args.crop_bottom
    dynamic_mask_cache_dir = args.dynamic_mask_cache_dir or args.sam2_cache_dir

    if args.sliding_window:
        scene_dir, rendered_paths, depth_paths, video_path, comparison_path = run_sliding_window_scene_pipeline(
            data_root=args.data_root,
            scene_id=args.scene_id,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            merge_frames=args.merge_frames,
            target_fps=args.target_fps,
            image_resolution=args.image_resolution,
            device=args.device,
            crop_bottom_pixels=crop_bottom,
            crop_cache_dir=args.crop_cache_dir,
            sky_mask_cache_dir=args.sky_mask_cache_dir,
            image_cache_dir=args.image_cache_dir,
            conf_percentile=args.conf_percentile,
            align_metric=not args.no_metric_alignment,
            save_comparison=not args.no_comparison_gif,
            comparison_fps=args.comparison_fps,
            filter_dynamic=not args.no_dynamic_filter,
            dynamic_filter_mode=args.dynamic_filter_mode,
            scale_error_threshold=args.scale_error_threshold,
            min_box_displacement_m=args.min_box_displacement_m,
            box_expand_ratio=args.box_filter_expand_ratio,
            max_lidar_points=args.max_lidar_prompt_points,
            sam2_model_id=args.sam2_model_id,
            dynamic_mask_cache_dir=dynamic_mask_cache_dir,
            skip_mask_precompute=args.skip_mask_precompute,
            debug_dynamic_filter=args.debug_dynamic_filter,
        )
        print(f"Saved scene outputs to {scene_dir}")
        print(f"  rgb:   {len(rendered_paths)} frames in {scene_dir / 'rgb'}")
        print(f"  depth: {len(depth_paths)} npz files in {scene_dir / 'depth'}")
        if args.debug_dynamic_filter:
            print(f"  dynamic debug: {scene_dir / 'dynamic_debug'}")
        if video_path is not None:
            print(f"  video: {video_path}")
        if comparison_path is not None:
            print(f"  comparison: {comparison_path}")
        return

    _, rendered_paths, comparison_path = run_render_pipeline(
        data_root=args.data_root,
        scene_id=args.scene_id,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        target_fps=args.target_fps,
        image_resolution=args.image_resolution,
        device=args.device,
        crop_bottom_pixels=crop_bottom,
        crop_cache_dir=args.crop_cache_dir,
        sky_mask_cache_dir=args.sky_mask_cache_dir,
        image_cache_dir=args.image_cache_dir,
        conf_percentile=args.conf_percentile,
        align_metric=not args.no_metric_alignment,
        save_comparison=not args.no_comparison_gif,
        comparison_fps=args.comparison_fps,
        filter_dynamic=not args.no_dynamic_filter,
        dynamic_filter_mode=args.dynamic_filter_mode,
        scale_error_threshold=args.scale_error_threshold,
        min_box_displacement_m=args.min_box_displacement_m,
        box_expand_ratio=args.box_filter_expand_ratio,
        sam2_model_id=args.sam2_model_id,
        sam2_cache_dir=args.sam2_cache_dir,
        debug_dynamic_filter=args.debug_dynamic_filter,
    )
    print(f"Saved {len(rendered_paths)} renders to {args.output_dir / args.scene_id}")
    if args.debug_dynamic_filter:
        print(f"Saved dynamic filter debug to {args.output_dir / args.scene_id / 'dynamic_debug'}")
    if comparison_path is not None:
        print(f"Saved comparison GIF to {comparison_path}")


if __name__ == "__main__":
    main()
