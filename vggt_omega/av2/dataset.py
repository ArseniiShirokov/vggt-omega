from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3
from av2.structures.sweep import Sweep
from av2.utils.io import read_ego_SE3_sensor, read_feather, read_img
from av2.utils.typing import NDArrayByte, NDArrayFloat

FRONT_CAMERA = RingCameras.RING_FRONT_CENTER.value
CAMERA_FPS = 20
DEFAULT_TARGET_FPS = 10
DEFAULT_AV2_CROP_BOTTOM = 205


def _se3_to_matrix(se3: SE3) -> NDArrayFloat:
    return se3.transform_matrix.copy()


def _box_at_camera_time(
    *,
    track_uuid: str,
    category: str,
    length_m: float,
    width_m: float,
    height_m: float,
    rotation: NDArrayFloat,
    translation: NDArrayFloat,
    city_SE3_ego_lidar,
    city_SE3_ego_cam,
) -> AV2Box3D:
    object_ego_lidar = SE3(rotation=rotation, translation=translation)
    ego_cam_SE3_ego_lidar = city_SE3_ego_cam.inverse().compose(city_SE3_ego_lidar)
    object_ego_cam = ego_cam_SE3_ego_lidar.compose(object_ego_lidar)
    return AV2Box3D(
        track_uuid=track_uuid,
        category=category,
        length_m=float(length_m),
        width_m=float(width_m),
        height_m=float(height_m),
        ego_SE3_object=_se3_to_matrix(object_ego_cam),
    )


def _load_boxes_by_lidar_timestamp(annotations_path: Path) -> dict[int, list[dict[str, object]]]:
    if not annotations_path.exists():
        return {}

    data = read_feather(annotations_path)
    rotations = quat_to_mat(data.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy())
    translations = data.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()

    boxes_by_timestamp: dict[int, list[dict[str, object]]] = {}
    for index, row in enumerate(data.itertuples(index=False)):
        boxes_by_timestamp.setdefault(int(row.timestamp_ns), []).append(
            {
                "track_uuid": str(row.track_uuid),
                "category": str(row.category),
                "length_m": float(row.length_m),
                "width_m": float(row.width_m),
                "height_m": float(row.height_m),
                "rotation": rotations[index],
                "translation": translations[index],
            }
        )
    return boxes_by_timestamp


@dataclass(frozen=True)
class AV2Box3D:
    """3D bounding box in the ego frame at camera capture time."""

    track_uuid: str
    category: str
    length_m: float
    width_m: float
    height_m: float
    ego_SE3_object: NDArrayFloat


@dataclass(frozen=True)
class AV2Frame:
    """Single front-camera frame with aligned LiDAR metadata."""

    log_id: str
    cam_timestamp_ns: int
    lidar_timestamp_ns: int
    image_path: Path
    lidar_path: Path
    intrinsics: NDArrayFloat
    ego_SE3_cam: NDArrayFloat
    ego_SE3_up_lidar: NDArrayFloat
    ego_SE3_down_lidar: NDArrayFloat
    city_SE3_ego: SE3
    boxes: tuple[AV2Box3D, ...] = ()


class AV2SceneDataset:
    """Front-camera frames and aligned LiDAR for one Argoverse 2 log."""

    def __init__(
        self,
        data_root: Path | str,
        log_id: str,
        *,
        target_fps: float = DEFAULT_TARGET_FPS,
        camera_name: str = FRONT_CAMERA,
        load_boxes: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.log_id = log_id
        self.target_fps = target_fps
        self.camera_name = camera_name

        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if target_fps > CAMERA_FPS:
            raise ValueError(
                f"target_fps must be <= {CAMERA_FPS} Hz for Argoverse 2 ring cameras"
            )

        self._loader = AV2SensorDataLoader(
            data_dir=self.data_root,
            labels_dir=self.data_root,
        )
        self._camera = self._loader.get_log_pinhole_camera(log_id, camera_name)
        sensor_poses = read_ego_SE3_sensor(log_dir=self.data_root / log_id)
        self._intrinsics = self._camera.intrinsics.K.copy()
        self._ego_SE3_cam = _se3_to_matrix(self._camera.ego_SE3_cam)
        self._ego_SE3_up_lidar = _se3_to_matrix(sensor_poses["up_lidar"])
        self._ego_SE3_down_lidar = _se3_to_matrix(sensor_poses["down_lidar"])

        self._cam_paths = self._subsample_camera_paths(
            self._loader.get_ordered_log_cam_fpaths(log_id, camera_name)
        )
        if len(self._cam_paths) == 0:
            raise ValueError(f"No camera frames found for log {log_id}")

        self._boxes_by_lidar_ts: dict[int, list[dict[str, object]]] = {}
        if load_boxes:
            self._boxes_by_lidar_ts = _load_boxes_by_lidar_timestamp(
                self.data_root / log_id / "annotations.feather"
            )

    @staticmethod
    def load_image(image_path: Path) -> NDArrayByte:
        return read_img(image_path, channel_order="RGB")

    @staticmethod
    def load_sweep(lidar_path: Path) -> Sweep:
        return Sweep.from_feather(lidar_path)

    def _subsample_camera_paths(self, cam_paths: Sequence[Path]) -> list[Path]:
        if len(cam_paths) <= 1:
            return list(cam_paths)

        stride = max(1, round(CAMERA_FPS / self.target_fps))
        return list(cam_paths[::stride])

    def _boxes_at_camera_time(
        self,
        lidar_timestamp_ns: int,
        city_SE3_ego_cam: SE3,
    ) -> tuple[AV2Box3D, ...]:
        rows = self._boxes_by_lidar_ts.get(lidar_timestamp_ns, [])
        if not rows:
            return ()

        city_SE3_ego_lidar = self._loader.get_city_SE3_ego(self.log_id, lidar_timestamp_ns)
        return tuple(
            _box_at_camera_time(
                track_uuid=row["track_uuid"],
                category=row["category"],
                length_m=row["length_m"],
                width_m=row["width_m"],
                height_m=row["height_m"],
                rotation=row["rotation"],
                translation=row["translation"],
                city_SE3_ego_lidar=city_SE3_ego_lidar,
                city_SE3_ego_cam=city_SE3_ego_cam,
            )
            for row in rows
        )

    def __len__(self) -> int:
        return len(self._cam_paths)

    def __getitem__(self, index: int) -> AV2Frame:
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index {index} out of range for log {self.log_id}")

        image_path = self._cam_paths[index]
        cam_timestamp_ns = int(image_path.stem)

        lidar_path = self._loader.get_closest_lidar_fpath(self.log_id, cam_timestamp_ns)
        if lidar_path is None:
            raise RuntimeError(
                f"No LiDAR sweep found for camera timestamp {cam_timestamp_ns} in log {self.log_id}"
            )

        lidar_timestamp_ns = int(lidar_path.stem)
        city_SE3_ego = self._loader.get_city_SE3_ego(self.log_id, cam_timestamp_ns)

        return AV2Frame(
            log_id=self.log_id,
            cam_timestamp_ns=cam_timestamp_ns,
            lidar_timestamp_ns=lidar_timestamp_ns,
            image_path=image_path,
            lidar_path=lidar_path,
            intrinsics=self._intrinsics,
            ego_SE3_cam=self._ego_SE3_cam,
            ego_SE3_up_lidar=self._ego_SE3_up_lidar,
            ego_SE3_down_lidar=self._ego_SE3_down_lidar,
            city_SE3_ego=city_SE3_ego,
            boxes=self._boxes_at_camera_time(lidar_timestamp_ns, city_SE3_ego),
        )

    def __iter__(self) -> Iterator[AV2Frame]:
        for index in range(len(self)):
            yield self[index]
