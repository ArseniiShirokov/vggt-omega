from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from vggt_omega.waymo.dataset import DEFAULT_WAYMO_CROP_BOTTOM, WaymoSceneDataset
from vggt_omega.waymo.metric_alignment import apply_metric_scale, compute_metric_scale_from_first_frame
from vggt_omega.av2.inference import (
    run_vggt_on_chunk,
    unproject_depth_map_to_point_map,
)


def crop_waymo_images(
    image_paths: list[str | Path],
    crop_bottom: int,
    cache_dir: Path,
) -> list[str]:
    if crop_bottom <= 0:
        return [str(path) for path in image_paths]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cropped_paths: list[str] = []
    for path in image_paths:
        path = Path(path)
        out_path = cache_dir / path.name
        if out_path.exists():
            cropped_paths.append(str(out_path))
            continue

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if crop_bottom >= height:
                raise ValueError(
                    f"crop_bottom={crop_bottom} exceeds image height={height} for {path}"
                )
            image.crop((0, 0, width, height - crop_bottom)).save(out_path, quality=95)
        cropped_paths.append(str(out_path))

    return cropped_paths


def run_vggt_on_waymo_chunk(
    data_root: str | Path,
    scene_id: str,
    frame_start: int,
    frame_end: int,
    checkpoint_path: str | Path,
    *,
    split: str = "training",
    target_fps: float = 10.0,
    image_resolution: int = 512,
    preprocess_mode: str = "max_size",
    device: str = "cuda",
    apply_sky_mask: bool = True,
    sky_mask_cache_dir: str | Path | None = None,
    skyseg_model_path: str = "skyseg.onnx",
    crop_bottom_pixels: int = DEFAULT_WAYMO_CROP_BOTTOM,
    crop_cache_dir: str | Path | None = None,
    align_metric: bool = True,
    metric_percentile: float = 90.0,
    image_cache_dir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], list[Path], list[str]]:
    """Run VGGT-Omega on an inclusive Waymo frame range [frame_start, frame_end]."""
    if frame_start < 0:
        raise ValueError("frame_start must be >= 0")
    if frame_end < frame_start:
        raise ValueError("frame_end must be >= frame_start")
    if crop_bottom_pixels < 0:
        raise ValueError("crop_bottom_pixels must be >= 0")

    scene = WaymoSceneDataset(
        data_root,
        scene_id,
        split=split,
        target_fps=target_fps,
        image_cache_dir=image_cache_dir,
    )
    if frame_end >= len(scene):
        raise IndexError(
            f"frame_end={frame_end} is out of range for scene {scene_id} with {len(scene)} frames"
        )

    frames = [scene[index] for index in range(frame_start, frame_end + 1)]
    image_paths = [frame.image_path for frame in frames]

    if crop_bottom_pixels > 0:
        if crop_cache_dir is None:
            crop_cache_dir = Path(tempfile.mkdtemp(prefix="waymo_crop_"))
        inference_paths = crop_waymo_images(image_paths, crop_bottom_pixels, Path(crop_cache_dir))
    else:
        inference_paths = [str(path) for path in image_paths]

    predictions = run_vggt_on_chunk(
        inference_paths,
        checkpoint_path,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
        device=device,
        apply_sky_mask=apply_sky_mask,
        sky_mask_cache_dir=sky_mask_cache_dir,
        skyseg_model_path=skyseg_model_path,
    )
    predictions["crop_bottom_pixels"] = np.array(crop_bottom_pixels)

    if align_metric:
        scale = compute_metric_scale_from_first_frame(
            predictions,
            frames[0],
            crop_bottom=crop_bottom_pixels,
            percentile=metric_percentile,
        )
        predictions = apply_metric_scale(predictions, scale)
        predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"],
            predictions["intrinsic"],
        )

    return predictions, image_paths, inference_paths
