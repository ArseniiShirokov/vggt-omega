from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_RENDER_PIPELINE_CONFIG = Path("configs/render_pipeline.yaml")


def load_render_pipeline_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline config must be a mapping: {config_path}")

    return _flatten_scene(raw)


def _flatten_scene(raw: dict[str, Any]) -> dict[str, Any]:
    flat = dict(raw)
    scene = flat.pop("scene", None)
    if scene is not None:
        if not isinstance(scene, dict):
            raise ValueError("'scene' section must be a mapping")
        for key in ("log_id", "frame_start", "frame_end"):
            if key in scene:
                flat[key] = scene[key]

    sliding = flat.pop("sliding_window", None)
    if sliding is not None:
        if not isinstance(sliding, dict):
            raise ValueError("'sliding_window' section must be a mapping")
        if "enabled" in sliding:
            flat["sliding_window"] = sliding["enabled"]
        if "merge_frames" in sliding:
            flat["merge_frames"] = sliding["merge_frames"]

    return flat


def config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Map config keys to argparse destination names."""
    return {
        "data_root": config.get("data_root"),
        "log_id": config.get("log_id"),
        "frame_start": config.get("frame_start"),
        "frame_end": config.get("frame_end"),
        "checkpoint": config.get("checkpoint"),
        "output_dir": config.get("output_dir"),
        "image_resolution": config.get("image_resolution"),
        "target_fps": config.get("target_fps"),
        "device": config.get("device"),
        "crop_bottom": config.get("crop_bottom"),
        "crop_cache_dir": config.get("crop_cache_dir"),
        "sky_mask_cache_dir": config.get("sky_mask_cache_dir"),
        "conf_percentile": config.get("conf_percentile"),
        "no_metric_alignment": not config.get("align_metric", True),
        "no_crop": config.get("no_crop", False),
        "no_comparison_gif": not config.get("save_comparison", True),
        "comparison_fps": config.get("comparison_fps"),
        "no_dynamic_filter": not config.get("dynamic_filter", True),
        "dynamic_filter_mode": config.get("dynamic_filter_mode"),
        "scale_error_threshold": config.get("scale_error_threshold"),
        "min_box_displacement_m": config.get("min_box_displacement_m"),
        "box_filter_expand_ratio": config.get("box_filter_expand_ratio"),
        "max_lidar_prompt_points": config.get("max_lidar_prompt_points"),
        "sam2_model_id": config.get("sam2_model_id"),
        "sam2_cache_dir": config.get("sam2_cache_dir"),
        "dynamic_mask_cache_dir": config.get("dynamic_mask_cache_dir") or config.get("sam2_cache_dir"),
        "debug_dynamic_filter": config.get("debug_dynamic_filter", False),
        "sliding_window": config.get("sliding_window", False),
        "merge_frames": config.get("merge_frames", 8),
        "skip_mask_precompute": config.get("skip_mask_precompute", False),
    }
