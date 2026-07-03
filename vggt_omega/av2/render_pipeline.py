"""Merge metric VGGT depth with AV2 calibrations and render camera views."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.utils.io import read_img
from PIL import Image

from vggt_omega.av2.dataset import AV2SceneDataset, DEFAULT_AV2_CROP_BOTTOM
from vggt_omega.av2.inference import run_vggt_on_av2_chunk
from vggt_omega.av2.metric_alignment import (
    build_pinhole_camera,
    depth_to_cam_points,
    motion_compensate_ego,
    scale_pinhole_camera,
)


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
    image = read_img(image_path, channel_order="RGB")
    height, width = image.shape[:2]
    u = np.clip(np.round(uv[:, 0] * width / camera_width).astype(np.int32), 0, width - 1)
    v = np.clip(np.round(uv[:, 1] * height / camera_height).astype(np.int32), 0, height - 1)
    return image[v, u]


def merge_depth_maps_in_ego0(
    predictions: dict[str, np.ndarray],
    frames: list,
    camera: PinholeCamera,
    color_image_paths: list[Path],
    *,
    conf_percentile: float = 0.0,
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

        points_ego = camera.ego_SE3_cam.transform_point_cloud(points_cam[keep])
        points_ego0 = motion_compensate_ego(points_ego, frame.city_SE3_ego, frame_0.city_SE3_ego)

        all_points_ego0.append(points_ego0)
        all_colors.append(
            sample_image_colors(
                color_image_paths[index],
                uv[keep],
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


def render_points_in_camera(
    camera: PinholeCamera,
    points_cam: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    image = np.zeros((camera.height_px, camera.width_px, 3), dtype=np.uint8)
    depth_buffer = np.full((camera.height_px, camera.width_px), np.inf, dtype=np.float32)

    uv, points_cam, valid = camera.project_cam_to_img(points_cam)
    uv = np.round(uv[valid]).astype(np.int32)
    z = points_cam[valid, 2]
    colors = colors[valid]

    u, v = uv[:, 0], uv[:, 1]
    in_bounds = (u >= 0) & (u < camera.width_px) & (v >= 0) & (v < camera.height_px) & (z > 0)
    u, v, z, colors = u[in_bounds], v[in_bounds], z[in_bounds], colors[in_bounds]

    for idx in np.argsort(-z):
        if z[idx] < depth_buffer[v[idx], u[idx]]:
            depth_buffer[v[idx], u[idx]] = z[idx]
            image[v[idx], u[idx]] = colors[idx]

    return image


def load_gt_image(
    image_path: Path,
    *,
    crop_bottom: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Load a GT camera frame cropped and resized to the render resolution."""
    image = read_img(image_path, channel_order="RGB")
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
    """Place GT and render side by side with a divider and labels."""
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
    """Build a side-by-side GT vs render GIF."""
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
    frames: list,
    camera: PinholeCamera,
    output_dir: Path,
) -> list[Path]:
    frame_0 = frames[0]
    cam_SE3_ego = camera.ego_SE3_cam.inverse()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []

    for index, frame in enumerate(frames):
        points_ego = motion_compensate_ego(merged.points_ego0, frame_0.city_SE3_ego, frame.city_SE3_ego)
        points_cam = cam_SE3_ego.transform_point_cloud(points_ego)
        rendered = render_points_in_camera(camera, points_cam, merged.colors)
        output_path = output_dir / f"{index:04d}_{frame.cam_timestamp_ns}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        rendered_paths.append(output_path)

    return rendered_paths


def run_render_pipeline(
    data_root: str | Path,
    log_id: str,
    frame_start: int,
    frame_end: int,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    target_fps: float = 10.0,
    image_resolution: int = 512,
    preprocess_mode: str = "max_size",
    device: str = "cuda",
    crop_bottom_pixels: int = DEFAULT_AV2_CROP_BOTTOM,
    crop_cache_dir: str | Path | None = None,
    sky_mask_cache_dir: str | Path | None = None,
    conf_percentile: float = 0.0,
    align_metric: bool = True,
    save_comparison: bool = True,
    comparison_fps: float | None = None,
) -> tuple[MergedPointCloud, list[Path], Path | None]:
    output_dir = Path(output_dir)
    data_root = Path(data_root)

    predictions, image_paths, inference_paths = run_vggt_on_av2_chunk(
        data_root=data_root,
        log_id=log_id,
        frame_start=frame_start,
        frame_end=frame_end,
        checkpoint_path=checkpoint_path,
        target_fps=target_fps,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
        device=device,
        apply_sky_mask=True,
        sky_mask_cache_dir=sky_mask_cache_dir,
        crop_bottom_pixels=crop_bottom_pixels,
        crop_cache_dir=crop_cache_dir,
        align_metric=align_metric,
    )

    scene = AV2SceneDataset(data_root, log_id, target_fps=target_fps)
    frames = [scene[index] for index in range(frame_start, frame_end + 1)]
    pred_height, pred_width = predictions["depth"].shape[1:3]
    camera = scale_pinhole_camera(
        build_pinhole_camera(data_root, frames[0], crop_bottom_pixels),
        pred_width,
        pred_height,
    )

    merged = merge_depth_maps_in_ego0(
        predictions,
        frames,
        camera,
        [Path(path) for path in inference_paths],
        conf_percentile=conf_percentile,
    )
    rendered_paths = render_all_camera_views(merged, frames, camera, output_dir / "renders")

    np.savez(
        output_dir / "merged_pointcloud.npz",
        points_ego0=merged.points_ego0,
        points_cam0=merged.points_cam0,
        colors=merged.colors,
    )
    np.savez(output_dir / "predictions.npz", **predictions)

    comparison_path: Path | None = None
    if save_comparison:
        comparison_path = save_comparison_gif(
            [Path(path) for path in image_paths],
            rendered_paths,
            output_dir / "comparison.gif",
            crop_bottom=crop_bottom_pixels,
            width=camera.width_px,
            height=camera.height_px,
            fps=target_fps if comparison_fps is None else comparison_fps,
        )

    return merged, rendered_paths, comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT AV2 inference, fusion, and render pipeline")
    parser.add_argument("--data-root", type=Path, default=Path("argoverse2"))
    parser.add_argument("--log-id", required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-bottom", type=int, default=DEFAULT_AV2_CROP_BOTTOM)
    parser.add_argument("--crop-cache-dir", type=Path, default=None)
    parser.add_argument("--sky-mask-cache-dir", type=Path, default=None)
    parser.add_argument("--conf-percentile", type=float, default=0.0)
    parser.add_argument("--no-metric-alignment", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-comparison-gif", action="store_true")
    parser.add_argument("--comparison-fps", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rendered_paths, comparison_path = run_render_pipeline(
        data_root=args.data_root,
        log_id=args.log_id,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        image_resolution=args.image_resolution,
        device=args.device,
        crop_bottom_pixels=0 if args.no_crop else args.crop_bottom,
        crop_cache_dir=args.crop_cache_dir,
        sky_mask_cache_dir=args.sky_mask_cache_dir,
        conf_percentile=args.conf_percentile,
        align_metric=not args.no_metric_alignment,
        save_comparison=not args.no_comparison_gif,
        comparison_fps=args.comparison_fps,
    )
    print(f"Saved {len(rendered_paths)} renders to {args.output_dir / 'renders'}")
    if comparison_path is not None:
        print(f"Saved comparison GIF to {comparison_path}")


if __name__ == "__main__":
    main()
