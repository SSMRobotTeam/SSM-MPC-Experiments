from dataclasses import dataclass

import numpy as np

from utils.quat_rotR import quat2dcm


@dataclass
class MotionReference:
    end_effector: np.ndarray
    mid_effector: np.ndarray
    base_xyz: np.ndarray
    base_to_global: np.ndarray


def _xyz(obj) -> np.ndarray:
    return np.array([obj.X, obj.Y, obj.Z], dtype=float)


def _local_position(point_xyz: np.ndarray, reference: MotionReference) -> np.ndarray:
    return (reference.base_to_global @ (point_xyz - reference.base_xyz).T).T


def initialize_motion_reference(
    sdk,
    base_rigid_id: int = 0,
    mid_rigid_id: int = 1,
    end_marker_id: int = 100028,
    base_y_offset: float = -12,
) -> MotionReference:
    sdk.ReceiveData(0)
    rigid_info = sdk.GetRigid()
    marker_info = sdk.GetMarker()

    base = rigid_info[base_rigid_id]
    base_xyz = np.array([base.X, base.Y - base_y_offset, base.Z], dtype=float)
    base_q = np.array([base.qw, base.qx, base.qy, base.qz], dtype=float)
    base_to_global = quat2dcm(base_q)

    reference = MotionReference(
        end_effector=np.zeros(3),
        mid_effector=np.zeros(3),
        base_xyz=base_xyz,
        base_to_global=base_to_global,
    )
    reference.mid_effector = _local_position(_xyz(rigid_info[mid_rigid_id]), reference)
    reference.end_effector = _local_position(_xyz(marker_info[end_marker_id]), reference)
    return reference


def capture_motion(
    sdk,
    reference: MotionReference,
    mid_rigid_id: int = 1,
    end_marker_id: int = 100028,
) -> tuple[np.ndarray, np.ndarray]:
    sdk.ReceiveData(0)
    rigid_info = sdk.GetRigid()
    marker_info = sdk.GetMarker()

    mid_position = _local_position(_xyz(rigid_info[mid_rigid_id]), reference)
    end_position = _local_position(_xyz(marker_info[end_marker_id]), reference)

    return (
        end_position - reference.end_effector,
        mid_position - reference.mid_effector,
    )
