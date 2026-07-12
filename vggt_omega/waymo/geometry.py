from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Waymo vehicle frame -> OpenCV camera frame (same as waymo/layers.py).
WAYMO_TO_CV = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SE3:
    """Rigid transform stored as a 4x4 homogeneous matrix."""

    transform_matrix: np.ndarray

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> SE3:
        return cls(transform_matrix=np.asarray(matrix, dtype=np.float64))

    @classmethod
    def identity(cls) -> SE3:
        return cls.from_matrix(np.eye(4))

    def inverse(self) -> SE3:
        rotation = self.transform_matrix[:3, :3]
        translation = self.transform_matrix[:3, 3]
        inv = np.eye(4, dtype=np.float64)
        inv[:3, :3] = rotation.T
        inv[:3, 3] = -rotation.T @ translation
        return SE3.from_matrix(inv)

    def compose(self, other: SE3) -> SE3:
        return SE3.from_matrix(self.transform_matrix @ other.transform_matrix)

    def transform_point_cloud(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        rotation = self.transform_matrix[:3, :3]
        translation = self.transform_matrix[:3, 3]
        return points @ rotation.T + translation


@dataclass(frozen=True)
class Intrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx_px, 0.0, self.cx_px],
                [0.0, self.fy_px, self.cy_px],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PinholeCamera:
    """Pinhole camera with extrinsics ego_SE3_cam (camera pose in ego frame)."""

    ego_SE3_cam: SE3
    intrinsics: Intrinsics
    cam_name: str = "FRONT"

    @property
    def width_px(self) -> int:
        return self.intrinsics.width_px

    @property
    def height_px(self) -> int:
        return self.intrinsics.height_px

    def project_cam_to_img(
        self, points_cam: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(points_cam) == 0:
            empty_uv = np.zeros((0, 2), dtype=np.float64)
            return empty_uv, points_cam, np.zeros(0, dtype=bool)

        z = points_cam[:, 2]
        valid = z > 1e-6
        uv = np.zeros((len(points_cam), 2), dtype=np.float64)
        k = self.intrinsics.K
        uv[valid, 0] = k[0, 0] * points_cam[valid, 0] / z[valid] + k[0, 2]
        uv[valid, 1] = k[1, 1] * points_cam[valid, 1] / z[valid] + k[1, 2]
        return uv, points_cam, valid

    def project_ego_to_img(
        self, points_ego: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam_se3_ego = self.ego_SE3_cam.inverse()
        points_cam = cam_se3_ego.transform_point_cloud(points_ego)
        return self.project_cam_to_img(points_cam)
