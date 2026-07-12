from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from vggt_omega.waymo.geometry import SE3, WAYMO_TO_CV
from vggt_omega.waymo.types import WAYMO_TYPE_NAMES, WaymoBox3D, WaymoFrame
from vggt_omega.waymo.lidar import (
    ALL_LIDAR_IDXS,
    SIDE_LIDAR_IDXS,
    TOP_LIDAR_IDX,
    load_lidar_xyz_ego,
    process_lidar_calib,
)

FRONT_CAMERA = "FRONT"
CAMERA_FPS = 10
DEFAULT_TARGET_FPS = 10.0
DEFAULT_WAYMO_CROP_BOTTOM = 0

CAMERA_NAME_TO_ID: dict[str, int] = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 4,
    "LEFT": 3,
    "RIGHT": 5,
}

CALIB_CAMERA_MAPPING: dict[int, int] = {
    1: 1,
    2: 2,
    3: 4,
    4: 3,
    5: 5,
}


def read_npz_row(path: Path, row_idx: int, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    data = np.load(path / f"row_{row_idx}.npz")
    if columns is None:
        return {key: data[key] for key in data.files}
    return {key: data[key] for key in columns if key in data.files}


def _se3_from_pose_rows(poses: pd.DataFrame) -> list[SE3]:
    matrices = np.stack(
        poses["[VehiclePoseComponent].world_from_vehicle.transform"].tolist(),
        axis=0,
    ).reshape(-1, 4, 4)
    return [SE3.from_matrix(matrix) for matrix in matrices]


def _process_camera_calib(calib_df: pd.DataFrame) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for _, cam_row in calib_df.iterrows():
        camera_name = int(cam_row["key.camera_name"])
        mapped_name = CALIB_CAMERA_MAPPING.get(camera_name, camera_name)
        fx, fy, cx, cy = [
            float(cam_row[f"[CameraCalibrationComponent].intrinsic.{name}"])
            for name in ("f_u", "f_v", "c_u", "c_v")
        ]
        extr = np.asarray(cam_row["[CameraCalibrationComponent].extrinsic.transform"], dtype=np.float64).reshape(4, 4)
        width = int(cam_row["[CameraCalibrationComponent].width"])
        height = int(cam_row["[CameraCalibrationComponent].height"])
        result[mapped_name] = {
            "intrs": (fx, fy, cx, cy),
            "width": width,
            "height": height,
            "ego_se3_cam": SE3.from_matrix(extr),
        }
    return result


def _waymo_box_to_ego_se3_object(
    center: np.ndarray,
    size: np.ndarray,
    heading: float,
) -> np.ndarray:
    c, s = float(np.cos(heading)), float(np.sin(heading))
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return transform


def _load_boxes_by_timestamp(labels_path: Path) -> dict[int, list[dict[str, object]]]:
    if not labels_path.is_file():
        return {}

    labels = pd.read_parquet(labels_path)
    if labels.empty:
        return {}

    boxes_by_timestamp: dict[int, list[dict[str, object]]] = {}
    for _, row in labels.iterrows():
        timestamp_us = int(row["key.frame_timestamp_micros"])
        center = np.array(
            [
                float(row["[LiDARBoxComponent].box.center.x"]),
                float(row["[LiDARBoxComponent].box.center.y"]),
                float(row["[LiDARBoxComponent].box.center.z"]),
            ],
            dtype=np.float64,
        )
        size = np.array(
            [
                float(row["[LiDARBoxComponent].box.size.x"]),
                float(row["[LiDARBoxComponent].box.size.y"]),
                float(row["[LiDARBoxComponent].box.size.z"]),
            ],
            dtype=np.float64,
        )
        heading = float(row["[LiDARBoxComponent].box.heading"])
        type_id = int(row["[LiDARBoxComponent].type"])
        track_id = str(row["key.laser_object_id"])
        boxes_by_timestamp.setdefault(timestamp_us, []).append(
            {
                "track_id": track_id,
                "type_id": type_id,
                "category": WAYMO_TYPE_NAMES.get(type_id, f"TYPE_{type_id}"),
                "length_m": float(size[0]),
                "width_m": float(size[1]),
                "height_m": float(size[2]),
                "ego_SE3_object": _waymo_box_to_ego_se3_object(center, size, heading),
            }
        )
    return boxes_by_timestamp


class WaymoSceneDataset:
    """Front-camera frames and aligned LiDAR for one Waymo scene."""

    IMAGE_COLUMNS = [
        "[CameraImageComponent].image",
        "[CameraImageComponent].pose.transform",
        "key.camera_name",
        "[CameraImageComponent].pose_timestamp",
    ]

    LIDAR_COLUMNS = [
        "[LiDARComponent].range_image_return1.shape",
        "[LiDARComponent].range_image_return1.values",
    ]

    POSE_COLUMNS = [
        "[LiDARPoseComponent].range_image_return1.shape",
        "[LiDARPoseComponent].range_image_return1.values",
    ]

    def __init__(
        self,
        data_root: Path | str,
        scene_id: str,
        *,
        split: str = "training",
        target_fps: float = DEFAULT_TARGET_FPS,
        camera_name: str = FRONT_CAMERA,
        image_cache_dir: Path | str | None = None,
        use_side_lidar: bool = True,
        load_boxes: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.scene_id = str(scene_id)
        self.target_fps = target_fps
        self.camera_name = camera_name
        self.use_side_lidar = use_side_lidar
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir is not None else None

        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if target_fps > CAMERA_FPS:
            raise ValueError(f"target_fps must be <= {CAMERA_FPS} Hz for Waymo")

        self._split_root = self.data_root / split
        self._scene_root = self._split_root / "camera_image" / self.scene_id
        if not self._scene_root.is_dir():
            raise FileNotFoundError(f"Waymo scene not found: {self._scene_root}")

        poses = pd.read_parquet(self._split_root / "vehicle_pose" / f"{self.scene_id}.parquet")
        self._world_se3_egos = _se3_from_pose_rows(poses)
        self._poses_ts = poses["key.frame_timestamp_micros"].to_numpy(dtype=np.int64)

        self._camera_calib = _process_camera_calib(
            pd.read_parquet(self._split_root / "camera_calibration" / f"{self.scene_id}.parquet")
        )
        self._lidar_calib = process_lidar_calib(
            pd.read_parquet(self._split_root / "lidar_calibration" / f"{self.scene_id}.parquet")
        )

        self._camera_id = CAMERA_NAME_TO_ID[camera_name]
        self._sample_indices = self._subsample_indices(list(range(len(self._world_se3_egos))))
        if not self._sample_indices:
            raise ValueError(f"No frames found for scene {self.scene_id}")

        camera_calib = self._camera_calib[self._camera_id]
        fx, fy, cx, cy = camera_calib["intrs"]
        self._image_width = int(camera_calib["width"])
        self._image_height = int(camera_calib["height"])
        self._intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._static_ego_se3_cam = camera_calib["ego_se3_cam"]

        self._boxes_by_timestamp: dict[int, list[dict[str, object]]] = {}
        if load_boxes:
            self._boxes_by_timestamp = _load_boxes_by_timestamp(
                self._split_root / "lidar_box" / f"{self.scene_id}.parquet"
            )

    def _boxes_at_sample(self, sample_idx: int) -> tuple[WaymoBox3D, ...]:
        timestamp_us = int(self._poses_ts[sample_idx])
        rows = self._boxes_by_timestamp.get(timestamp_us, [])
        return tuple(
            WaymoBox3D(
                track_id=row["track_id"],
                type_id=row["type_id"],
                category=row["category"],
                length_m=row["length_m"],
                width_m=row["width_m"],
                height_m=row["height_m"],
                ego_SE3_object=row["ego_SE3_object"],
            )
            for row in rows
        )

    def _subsample_indices(self, indices: list[int]) -> list[int]:
        if len(indices) <= 1:
            return indices
        stride = max(1, round(CAMERA_FPS / self.target_fps))
        return indices[::stride]

    def __len__(self) -> int:
        return len(self._sample_indices)

    def is_usable(self, index: int) -> bool:
        if index < 0 or index >= len(self):
            return False
        sample_idx = self._sample_indices[index]
        lidar_path = self._split_root / "lidar" / self.scene_id
        row_idx = sample_idx * len(ALL_LIDAR_IDXS)
        return (lidar_path / f"row_{row_idx}.npz").is_file()

    def usable_indices(self, start: int, end: int) -> list[int]:
        return [index for index in range(start, end + 1) if self.is_usable(index)]

    def _image_row_idx(self, sample_idx: int) -> int:
        return sample_idx * len(CAMERA_NAME_TO_ID) + self._camera_id - 1

    def _extract_image(self, sample_idx: int) -> Path:
        if self.image_cache_dir is not None:
            out_path = self.image_cache_dir / f"{sample_idx:06d}.jpg"
            if out_path.is_file():
                return out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = self._split_root / "cache" / self.scene_id / "images" / f"{sample_idx:06d}.jpg"
            if out_path.is_file():
                return out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

        row = read_npz_row(self._scene_root, self._image_row_idx(sample_idx), self.IMAGE_COLUMNS)
        image = Image.open(io.BytesIO(row["[CameraImageComponent].image"])).convert("RGB")
        image.save(out_path, quality=95)
        return out_path

    def _load_lidar_xyz_ego(self, sample_idx: int, *, use_side_lidar: bool | None = None) -> np.ndarray:
        lidar_root = self._split_root / "lidar" / self.scene_id
        pose_root = self._split_root / "lidar_pose" / self.scene_id
        world_se3_ego = self._world_se3_egos[sample_idx]
        include_side = self.use_side_lidar if use_side_lidar is None else use_side_lidar

        top_row_idx = sample_idx * len(ALL_LIDAR_IDXS)
        top_row = read_npz_row(lidar_root, top_row_idx, self.LIDAR_COLUMNS)
        pose_row = read_npz_row(pose_root, sample_idx, self.POSE_COLUMNS)

        side_rows: list[tuple[int, dict[str, np.ndarray]]] = []
        if include_side:
            for lidar_idx in SIDE_LIDAR_IDXS:
                row_idx = sample_idx * len(ALL_LIDAR_IDXS) + lidar_idx - 1
                side_rows.append((lidar_idx, read_npz_row(lidar_root, row_idx, self.LIDAR_COLUMNS)))

        return load_lidar_xyz_ego(
            self._lidar_calib,
            top_row,
            pose_row,
            world_se3_ego,
            use_side_lidar=include_side,
            side_rows=side_rows,
        )

    def image_path_at(self, index: int) -> Path:
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index {index} out of range for scene {self.scene_id}")
        return self._extract_image(self._sample_indices[index])

    def _ego_se3_cam_at_sample(self, sample_idx: int) -> np.ndarray:
        row = read_npz_row(
            self._scene_root,
            self._image_row_idx(sample_idx),
            ["[CameraImageComponent].pose.transform"],
        )
        world_se3_ego_image = SE3.from_matrix(
            np.asarray(row["[CameraImageComponent].pose.transform"], dtype=np.float64).reshape(4, 4)
        )
        world_se3_ego_frame = self._world_se3_egos[sample_idx]
        ego_se3_cam = world_se3_ego_frame.inverse().compose(world_se3_ego_image).compose(
            self._static_ego_se3_cam
        )
        ego_se3_cam_cv = ego_se3_cam.compose(SE3.from_matrix(WAYMO_TO_CV))
        return ego_se3_cam_cv.transform_matrix

    def get_frame(
        self,
        index: int,
        *,
        load_lidar: bool = True,
        use_side_lidar: bool | None = None,
    ) -> WaymoFrame:
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index {index} out of range for scene {self.scene_id}")

        sample_idx = self._sample_indices[index]
        lidar_xyz = (
            self._load_lidar_xyz_ego(sample_idx, use_side_lidar=use_side_lidar)
            if load_lidar
            else np.zeros((0, 3), dtype=np.float64)
        )
        return WaymoFrame(
            scene_id=self.scene_id,
            sample_idx=sample_idx,
            timestamp_us=int(self._poses_ts[sample_idx]),
            image_path=self._extract_image(sample_idx),
            intrinsics=self._intrinsics.copy(),
            ego_SE3_cam=self._ego_se3_cam_at_sample(sample_idx),
            world_SE3_ego=self._world_se3_egos[sample_idx],
            lidar_xyz_ego=lidar_xyz,
            image_width=self._image_width,
            image_height=self._image_height,
            boxes=self._boxes_at_sample(sample_idx),
        )

    def __getitem__(self, index: int) -> WaymoFrame:
        return self.get_frame(index, load_lidar=True)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
