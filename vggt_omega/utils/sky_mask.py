# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import numpy as np
import requests


def apply_sky_mask_from_images(
    conf: np.ndarray,
    images: np.ndarray,
    *,
    skyseg_model_path: str = "skyseg.onnx",
) -> np.ndarray:
    """Apply sky segmentation on the exact images used for VGGT inference."""
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 2:
        conf = conf[np.newaxis]

    num_frames, height, width = conf.shape
    if images.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dimensions, got shape {images.shape}")

    if images.shape[0] != num_frames:
        raise ValueError(
            f"Expected {num_frames} images for sky masking, got batch size {images.shape[0]}."
        )

    skyseg_session = _load_skyseg_session(skyseg_model_path)
    masks = []
    for index in range(num_frames):
        image_bgr = _prediction_image_to_bgr(images[index], height, width)
        masks.append(_segment_sky_array(image_bgr, skyseg_session))

    return _apply_sky_masks_to_conf(conf, masks)


def apply_sky_mask_from_paths(
    conf: np.ndarray,
    image_paths: list[str | os.PathLike],
    cache_dir: str | os.PathLike | None = None,
    skyseg_model_path: str = "skyseg.onnx",
) -> np.ndarray:
    """Apply sky segmentation masks to confidence maps using source image paths."""
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 2:
        conf = conf[np.newaxis]

    num_frames, height, width = conf.shape
    if len(image_paths) != num_frames:
        raise ValueError(
            f"Expected {num_frames} image paths for sky masking, got {len(image_paths)}."
        )

    skyseg_session = _load_skyseg_session(skyseg_model_path)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    masks = []
    for image_path in image_paths:
        image_path = str(image_path)
        image_name = os.path.basename(image_path)
        if cache_dir is not None:
            mask_path = os.path.join(cache_dir, image_name)
            if os.path.exists(mask_path):
                sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                sky_mask = segment_sky(image_path, skyseg_session, mask_path)
        else:
            image = cv2.imread(image_path)
            sky_mask = _segment_sky_array(image, skyseg_session)

        if sky_mask.shape != (height, width):
            sky_mask = cv2.resize(sky_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(sky_mask)

    return _apply_sky_masks_to_conf(conf, masks)


def segment_sky(image_path: str, onnx_session, mask_filename: str) -> np.ndarray:
    image = cv2.imread(image_path)
    output_mask = _segment_sky_array(image, onnx_session)

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def _load_skyseg_session(skyseg_model_path: str = "skyseg.onnx"):
    if not os.path.exists(skyseg_model_path):
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
            skyseg_model_path,
        )

    import onnxruntime

    return onnxruntime.InferenceSession(skyseg_model_path)


def _prediction_image_to_bgr(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[:2] != (height, width):
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match confidence shape {(height, width)}."
        )

    if image.max() <= 1.0:
        image = (image * 255.0).clip(0, 255)

    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _segment_sky_array(image_bgr: np.ndarray, onnx_session) -> np.ndarray:
    result_map = run_skyseg(onnx_session, [320, 320], image_bgr)
    result_map = cv2.resize(
        result_map,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    sky_mask = np.zeros_like(result_map)
    sky_mask[result_map < 32] = 255  # 255 = valid (non-sky) depth
    return sky_mask


def _apply_sky_masks_to_conf(conf: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Zero confidence on sky pixels. Cached masks use 255 for non-sky (valid depth)."""
    valid_depth = np.array(masks) > 0.1
    return conf * valid_depth.astype(np.float32)


def run_skyseg(onnx_session, input_size: list[int], image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.float32)
    image = (image / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = image.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: image})
    result = np.array(result).squeeze()
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)
    else:
        result = np.zeros_like(result)
    return (result * 255).astype("uint8")


def download_file_from_url(url: str, filename: str) -> None:
    tmp_filename = f"{filename}.tmp"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(tmp_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_filename, filename)
