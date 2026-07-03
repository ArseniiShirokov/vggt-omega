from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt_omega.utils.sky_mask import apply_sky_mask_from_images
from vggt_omega.av2.dataset import DEFAULT_AV2_CROP_BOTTOM
from vggt_omega.av2 import AV2SceneDataset
from vggt_omega.av2.metric_alignment import apply_metric_scale, compute_metric_scale_from_first_frame
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def crop_av2_images(
    image_paths: list[str | Path],
    crop_bottom: int,
    cache_dir: Path,
) -> list[str]:
    """Crop pixels from the bottom of AV2 front-camera images and cache the results."""
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
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            image = image.convert("RGB")
            width, height = image.size
            if crop_bottom >= height:
                raise ValueError(
                    f"crop_bottom={crop_bottom} exceeds image height={height} for {path}"
                )
            image.crop((0, 0, width, height - crop_bottom)).save(out_path, quality=95)

        cropped_paths.append(str(out_path))

    return cropped_paths


def load_model(checkpoint_path: str | Path, device: str = "cuda") -> VGGTOmega:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.to(device)


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def _predictions_to_numpy(predictions: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    predictions_np: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().float().cpu().numpy()
        if array.shape[0] == 1:
            array = array[0]
        predictions_np[key] = array
    return predictions_np


def run_vggt_on_chunk(
    image_paths: list[str | Path],
    checkpoint_path: str | Path,
    *,
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    device: str = "cuda",
    apply_sky_mask: bool = True,
    sky_mask_cache_dir: str | Path | None = None,
    skyseg_model_path: str = "skyseg.onnx",
) -> dict[str, np.ndarray]:
    """Run VGGT-Omega on a sequence of images and optionally filter sky in depth confidence."""
    if len(image_paths) == 0:
        raise ValueError("At least one image is required for inference")

    image_paths = [str(path) for path in image_paths]
    model = load_model(checkpoint_path, device=device)

    images = load_and_preprocess_images(
        image_paths,
        mode=preprocess_mode,
        image_resolution=image_resolution,
    ).to(device)
    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = _predictions_to_numpy(predictions)
    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    if apply_sky_mask:
        predictions_np["depth_conf"] = apply_sky_mask_from_images(
            predictions_np["depth_conf"],
            predictions_np["images"],
            skyseg_model_path=skyseg_model_path,
        )

    return predictions_np


def run_vggt_on_av2_chunk(
    data_root: str | Path,
    log_id: str,
    frame_start: int,
    frame_end: int,
    checkpoint_path: str | Path,
    *,
    target_fps: float = 10.0,
    image_resolution: int = 512,
    preprocess_mode: str = "max_size",
    device: str = "cuda",
    apply_sky_mask: bool = True,
    sky_mask_cache_dir: str | Path | None = None,
    skyseg_model_path: str = "skyseg.onnx",
    crop_bottom_pixels: int = DEFAULT_AV2_CROP_BOTTOM,
    crop_cache_dir: str | Path | None = None,
    align_metric: bool = True,
    metric_percentile: float = 90.0,
) -> tuple[dict[str, np.ndarray], list[Path], list[str]]:
    """Run VGGT-Omega on an inclusive AV2 frame range [frame_start, frame_end]."""
    if frame_start < 0:
        raise ValueError("frame_start must be >= 0")
    if frame_end < frame_start:
        raise ValueError("frame_end must be >= frame_start")
    if crop_bottom_pixels < 0:
        raise ValueError("crop_bottom_pixels must be >= 0")

    scene = AV2SceneDataset(data_root, log_id, target_fps=target_fps)
    if frame_end >= len(scene):
        raise IndexError(
            f"frame_end={frame_end} is out of range for log {log_id} with {len(scene)} frames"
        )

    frames = [scene[index] for index in range(frame_start, frame_end + 1)]
    image_paths = [frame.image_path for frame in frames]

    if crop_bottom_pixels > 0:
        if crop_cache_dir is None:
            crop_cache_dir = Path(tempfile.mkdtemp(prefix="av2_crop_"))
        inference_paths = crop_av2_images(image_paths, crop_bottom_pixels, Path(crop_cache_dir))
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
            data_root=data_root,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega on an Argoverse 2 frame chunk")
    parser.add_argument("--data-root", type=Path, default=Path("argoverse2"))
    parser.add_argument("--log-id", required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to save predictions.npz")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sky-mask-cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache sky segmentation masks",
    )
    parser.add_argument("--skyseg-model-path", default="skyseg.onnx")
    parser.add_argument("--no-sky-mask", action="store_true")
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=DEFAULT_AV2_CROP_BOTTOM,
        help="Crop this many pixels from the bottom before inference (default: 200)",
    )
    parser.add_argument(
        "--crop-cache-dir",
        type=Path,
        default=None,
        help="Directory to cache cropped AV2 images",
    )
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-metric-alignment", action="store_true")
    parser.add_argument("--metric-percentile", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, image_paths, inference_paths = run_vggt_on_av2_chunk(
        data_root=args.data_root,
        log_id=args.log_id,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        checkpoint_path=args.checkpoint,
        target_fps=args.target_fps,
        image_resolution=args.image_resolution,
        device=args.device,
        apply_sky_mask=not args.no_sky_mask,
        sky_mask_cache_dir=args.sky_mask_cache_dir,
        skyseg_model_path=args.skyseg_model_path,
        crop_bottom_pixels=0 if args.no_crop else args.crop_bottom,
        crop_cache_dir=args.crop_cache_dir,
        align_metric=not args.no_metric_alignment,
        metric_percentile=args.metric_percentile,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **predictions,
        image_paths=np.array([str(path) for path in image_paths]),
        inference_image_paths=np.array(inference_paths),
    )
    print(
        f"Saved predictions for frames [{args.frame_start}, {args.frame_end}] "
        f"({len(image_paths)} images) to {args.output}"
    )


if __name__ == "__main__":
    main()
