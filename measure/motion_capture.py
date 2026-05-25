from dataclasses import dataclass, field

import numpy as np

from utils.quat_rotR import quat2dcm


@dataclass
class MotionReference:
    end_effector: np.ndarray
    mid_effector: np.ndarray
    base_xyz: np.ndarray
    base_to_global: np.ndarray
    checkpoint: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obstacles: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


def _xyz(obj) -> np.ndarray:
    return np.array([obj.X, obj.Y, obj.Z], dtype=float)


def _item_id(item):
    for attr in ("rigidID", "RigidID", "rigidId", "RigidId", "markerID", "MarkerID", "id", "ID"):
        if hasattr(item, attr):
            return getattr(item, attr)
    return None


def _available_item_ids(items):
    if hasattr(items, "keys"):
        return list(items.keys())
    return [_item_id(item) for item in items]


def _item_by_id(items, item_id):
    keys_to_try = (item_id, int(item_id), str(item_id))
    if hasattr(items, "keys"):
        for key in keys_to_try:
            try:
                return items[key]
            except (KeyError, TypeError, ValueError):
                pass

    for item in items.values() if hasattr(items, "values") else items:
        value = _item_id(item)
        if value == item_id or str(value) == str(item_id):
            return item

    available = ", ".join(str(value) for value in _available_item_ids(items))
    raise KeyError(f"Could not find motion-capture item id {item_id}. Available ids: [{available}]")


def _local_position(point_xyz: np.ndarray, reference: MotionReference) -> np.ndarray:
    return (reference.base_to_global @ (point_xyz - reference.base_xyz).T).T


def initialize_motion_reference(
    sdk,
    base_rigid_id: int = 0,
    mid_rigid_id: int = 1,
    checkpoint_rigid_id: int = 2,
    end_marker_id: int = 100028,
    base_y_offset: float = -12,
) -> MotionReference:
    sdk.ReceiveData(0)
    rigid_info = sdk.GetRigid()
    marker_info = sdk.GetMarker()

    base = _item_by_id(rigid_info, base_rigid_id)
    base_xyz = np.array([base.X, base.Y - base_y_offset, base.Z], dtype=float)
    base_q = np.array([base.qw, base.qx, base.qy, base.qz], dtype=float)
    base_to_global = quat2dcm(base_q)

    reference = MotionReference(
        end_effector=np.zeros(3),
        mid_effector=np.zeros(3),
        base_xyz=base_xyz,
        base_to_global=base_to_global,
    )
    reference.mid_effector = _local_position(_xyz(_item_by_id(rigid_info, mid_rigid_id)), reference)
    reference.checkpoint = _local_position(_xyz(_item_by_id(rigid_info, checkpoint_rigid_id)), reference)
    reference.end_effector = _local_position(_xyz(_item_by_id(marker_info, end_marker_id)), reference)
    return reference


def capture_motion(
    sdk,
    reference: MotionReference,
    mid_rigid_id: int = 1,
    checkpoint_rigid_id: int = 2,
    end_marker_id: int = 100028,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdk.ReceiveData(0)
    rigid_info = sdk.GetRigid()
    marker_info = sdk.GetMarker()

    mid_position = _local_position(_xyz(_item_by_id(rigid_info, mid_rigid_id)), reference)
    checkpoint_position = _local_position(_xyz(_item_by_id(rigid_info, checkpoint_rigid_id)), reference)
    end_position = _local_position(_xyz(_item_by_id(marker_info, end_marker_id)), reference)

    return (
        end_position - reference.end_effector,
        mid_position - reference.mid_effector,
        checkpoint_position - reference.checkpoint,
    )


def capture_obstacle_positions(
    sdk,
    reference: MotionReference,
    obstacle_rigid_ids=(3, 4, 5),
) -> np.ndarray:
    sdk.ReceiveData(0)
    rigid_info = sdk.GetRigid()

    obstacle_positions = []
    for rigid_id in np.asarray(obstacle_rigid_ids, dtype=int).reshape(-1):
        obstacle_positions.append(_local_position(_xyz(_item_by_id(rigid_info, int(rigid_id))), reference))
    return np.asarray(obstacle_positions, dtype=float).reshape(-1, 3)


def capture_obstacle_positions_in_model_frame(
    sdk,
    reference: MotionReference,
    obstacle_rigid_ids=(3, 4, 5),
    pos_end=np.array([0.0, 0.0, 0.4400]),
) -> np.ndarray:
    obstacle_positions = capture_obstacle_positions(sdk, reference, obstacle_rigid_ids)
    pos_end = np.asarray(pos_end, dtype=float).reshape(3)
    return ((obstacle_positions - reference.end_effector.reshape(1, 3)) / 1000.0) + pos_end.reshape(1, 3)
