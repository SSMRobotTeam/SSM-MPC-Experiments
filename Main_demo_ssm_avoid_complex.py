#!/usr/bin/python
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import numpy as np
import scipy.io as sciIO

from controller.cable_constraints import cable_length_double_part
from controller.nmpc import nmpc_multi_shooting_zoh_avoid_complex
from controller.ssm_controller import SsmExperimentConfig, SsmFeedbackState, SsmMpcController
from dynamics.SSM.matlab_data import (
    EIGHT_TRAJ_DIR,
    SPIRAL_TRAJ_DIR,
    clone_struct_array,
    get_field,
    load_rom_data,
    subset_mapping_rows,
)
from measure.motion_capture import MotionReference, capture_motion, capture_obstacle_positions_in_model_frame
from utils.quat_rotR import quat2dcm
from utils.sdk_utils import load_lumo_sdk
from utils.timing_utils import fixed_cpu_time
from zmq_message.zmq_control import close_publisher, create_publisher, send_motor_command


BASE_DIR = Path(__file__).resolve().parent
MODEL_INFO_DIR = BASE_DIR / "dynamics" / "modelinfom"
PLOT_DIR = BASE_DIR / "PlotFig"
AVOID_COMPLEX_RESULT_DIR = PLOT_DIR / "Avoid_Complex"
RECORDED_CONTROL_INPUT_PATH = (
    AVOID_COMPLEX_RESULT_DIR / "05End_ok" / "ut_save_ssm_avoid_complex.npy"
)
LOG_DIR = PLOT_DIR / "logs"
OBSTACLE_INFO_DIR = BASE_DIR / "obstacleInformation"
OBSTACLE_INFO_PATH = OBSTACLE_INFO_DIR / "obstacle_information_complex.npz"
IP_ADDRESS = "127.0.0.1"
MOTOR_ENDPOINT = "tcp://localhost:5555"
CONTROL_PERIOD_SECONDS = 0.075
TRAJECTORY_NAME = "spiral"
SSM_FACTOR = 1.475
MOTOR_FACTOR = 1.225
horizon = 3
RUN_STEPS = np.inf
FEEDBACK_ENABLED = False
CONTROL_SOURCE = "online"  # choose one: "dataset" or "online"
MPC_SOLVER_TYPE = "ipopt"  # choose one: "ipopt" or "qp"
QP_BACKEND = "qpoases"  # used only when MPC_SOLVER_TYPE = "qp"
USE_FAST_SINGLE_STEP_QP = False
CONTROL_SMOOTH_WEIGHT = 2.5e4
Q_WEIGHT = 0.1 * np.diag([1e5, 1e5, 1e5, 1e3, 1e3, 1e3])
# Q_WEIGHT = 0.3 * np.diag([1e5, 1e5, 1e5, 1e3, 1e3, 1e3])
R_WEIGHT = 7.5e1
# R_WEIGHT = 1e1
SAVE_FEEDBACK_EACH_STEP = False
CAPTURE_MOTION_ENABLED = False
END_MARKER_ID = 100697 # 100027 
BASE_RIGID_ID = 0
MID_RIGID_ID = 1
FEEDBACK_RIGID_ID = 2
OBSTACLE_RIGID_IDS = (4,) # (3, 4, 5) # (3, 4, 5)  # obstacle rigid bodies measured by motion_capture
BASE_Y_OFFSET = -12.0
POS_END = np.array([0.0, 0.0, 0.4400])
OBSTACLE_RADIUS = 0.015
SAFE_DISTANCE_FACTOR = 2.0
OUTPUT_SUFFIX = "_complex"
# "node" keeps the existing obstacle-nearest/proximal node avoidance. "endeff"
# constrains the end-effector against each obstacle instead.
AVOID_CONSTRAINT_TARGET = "endeff"  # "node" or "endeff"
# "nearest" selects the closest model node to each obstacle. "proximal" first
# limits candidates to the base-side of the obstacle, making avoidance happen
# from a more proximal part of the trunk.
AVOID_NODE_SELECTION = "nearest"  # "nearest" or "proximal"
AVOID_PROXIMAL_OFFSET_M = 0.01
TRAJECTORY_DIRS = {
    "eight": EIGHT_TRAJ_DIR,
    "spiral": SPIRAL_TRAJ_DIR,
}


class BufferedTeeLogger:
    def __init__(self, stream):
        self.stream = stream
        self.messages = []

    def write(self, message):
        self.stream.write(message)
        self.messages.append(message)

    def flush(self):
        self.stream.flush()

    def save(self, log_dir: Path) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        beijing_now = datetime.now(timezone(timedelta(hours=8)))
        log_path = log_dir / f"SSM_avoid{OUTPUT_SUFFIX}_log_{beijing_now:%Y%m%d_%H%M}.txt"
        if log_path.exists():
            index = 2
            while True:
                candidate = log_dir / f"SSM_avoid{OUTPUT_SUFFIX}_log_{beijing_now:%Y%m%d_%H%M}_{index:02d}.txt"
                if not candidate.exists():
                    log_path = candidate
                    break
                index += 1
        log_path.write_text("".join(self.messages), encoding="utf-8")
        return log_path


@dataclass
class AvoidanceReference:
    motion: MotionReference
    feedback_point: np.ndarray
    obstacle_positions: np.ndarray
    obstacle_node_indices_1based: np.ndarray
    obstacle_node_positions: np.ndarray
    obstacle_node_full_dofs_0based: np.ndarray


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


def constrained_node_table(rom_data) -> tuple[np.ndarray, np.ndarray]:
    nodes_coord = np.asarray(rom_data["nodes_coord"], dtype=float)
    if nodes_coord.ndim != 2 or nodes_coord.shape[0] != 3:
        raise ValueError(f"Expected constrained nodes_coord shape (3, N), got {nodes_coord.shape}")
    return np.arange(nodes_coord.shape[1], dtype=int), nodes_coord


def endeff_obstacle_info(
    rom_data,
    obstacle_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obstacle_count = np.asarray(obstacle_positions, dtype=float).reshape(-1, 3).shape[0]
    outdof_endeff = np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)
    if outdof_endeff.size != 3:
        raise ValueError(f"Expected outdof_endeff to contain 3 DOFs, got {outdof_endeff.size}.")

    endeff_full_dofs_0based = outdof_endeff - 1
    endeff_node_indices = np.unique(endeff_full_dofs_0based // 3)
    node_index_1based = int(endeff_node_indices[0] + 1) if endeff_node_indices.size == 1 else 0
    return (
        np.full(obstacle_count, node_index_1based, dtype=int),
        np.repeat(POS_END.reshape(1, 3), obstacle_count, axis=0),
        np.tile(endeff_full_dofs_0based, obstacle_count),
    )


def obstacle_node_info(
    rom_data,
    obstacle_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_mode = str(AVOID_CONSTRAINT_TARGET).lower()
    if target_mode == "endeff":
        return endeff_obstacle_info(rom_data, obstacle_positions)
    if target_mode != "node":
        raise ValueError("AVOID_CONSTRAINT_TARGET must be 'node' or 'endeff'.")

    valid_node_indices, valid_nodes_coord = constrained_node_table(rom_data)
    positions = np.asarray(obstacle_positions, dtype=float).reshape(-1, 3)
    selection_mode = str(AVOID_NODE_SELECTION).lower()
    if selection_mode not in ("nearest", "proximal"):
        raise ValueError("AVOID_NODE_SELECTION must be 'nearest' or 'proximal'.")

    axial_coordinates = valid_nodes_coord[2, :]
    node_indices = []
    node_positions = []
    full_dofs = []

    for obstacle_position in positions:
        candidate_mask = np.ones(valid_nodes_coord.shape[1], dtype=bool)
        if selection_mode == "proximal":
            proximal_limit = float(obstacle_position[2]) - float(AVOID_PROXIMAL_OFFSET_M)
            candidate_mask = axial_coordinates <= proximal_limit
            if not np.any(candidate_mask):
                candidate_mask = axial_coordinates <= float(obstacle_position[2])
            if not np.any(candidate_mask):
                candidate_mask = np.ones(valid_nodes_coord.shape[1], dtype=bool)

        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_nodes = valid_nodes_coord[:, candidate_indices]
        distances = np.linalg.norm(candidate_nodes - obstacle_position.reshape(3, 1), axis=0)
        table_idx = int(candidate_indices[int(np.argmin(distances))])
        full_node_idx = int(valid_node_indices[table_idx])
        node_indices.append(full_node_idx + 1)
        node_positions.append(valid_nodes_coord[:, table_idx])
        full_dofs.extend([3 * full_node_idx, 3 * full_node_idx + 1, 3 * full_node_idx + 2])

    return (
        np.asarray(node_indices, dtype=int),
        np.asarray(node_positions, dtype=float),
        np.asarray(full_dofs, dtype=int),
    )


def load_obstacle_information(rom_data):
    if not OBSTACLE_INFO_PATH.exists():
        return None

    expected_node_count = int(np.asarray(rom_data["nodes_coord"], dtype=float).shape[1])
    try:
        data = np.load(OBSTACLE_INFO_PATH, allow_pickle=True)
        obstacle_positions = np.asarray(data["obstacle_positions"], dtype=float).reshape(-1, 3)
        obstacle_node_indices = np.asarray(data["obstacle_node_indices_1based"], dtype=int).reshape(-1)
        obstacle_node_positions = np.asarray(data["obstacle_node_positions"], dtype=float).reshape(-1, 3)
        obstacle_node_full_dofs = np.asarray(data["obstacle_node_full_dofs_0based"], dtype=int).reshape(-1)
        node_count = int(np.asarray(data["nodes_coord_constrained_count"]).reshape(-1)[0])
        if "obstacle_rigid_ids" in data.files:
            obstacle_rigid_ids = np.asarray(data["obstacle_rigid_ids"], dtype=int).reshape(-1)
        else:
            obstacle_rigid_ids = np.empty((0,), dtype=int)
        if "avoid_constraint_target" in data.files:
            cached_constraint_target = str(np.asarray(data["avoid_constraint_target"]).reshape(-1)[0]).lower()
        else:
            cached_constraint_target = ""
        if "avoid_node_selection" in data.files:
            cached_node_selection = str(np.asarray(data["avoid_node_selection"]).reshape(-1)[0]).lower()
        else:
            cached_node_selection = ""
        if "avoid_proximal_offset_m" in data.files:
            cached_proximal_offset_m = float(np.asarray(data["avoid_proximal_offset_m"]).reshape(-1)[0])
        else:
            cached_proximal_offset_m = np.nan
    except Exception as exc:
        print(f"Failed to load cached obstacle information from {OBSTACLE_INFO_PATH}: {exc}")
        return None

    obstacle_count = obstacle_positions.shape[0]
    expected_obstacle_ids = np.asarray(OBSTACLE_RIGID_IDS, dtype=int).reshape(-1)
    if (
        obstacle_count != expected_obstacle_ids.size
        or obstacle_node_indices.size != obstacle_count
        or obstacle_node_positions.shape != (obstacle_count, 3)
        or obstacle_node_full_dofs.size != 3 * obstacle_count
        or node_count != expected_node_count
        or np.max(obstacle_node_full_dofs, initial=-1) >= 3 * expected_node_count
        or obstacle_rigid_ids.size != expected_obstacle_ids.size
        or not np.array_equal(obstacle_rigid_ids, expected_obstacle_ids)
        or cached_constraint_target != str(AVOID_CONSTRAINT_TARGET).lower()
        or cached_node_selection != str(AVOID_NODE_SELECTION).lower()
        or not np.isclose(cached_proximal_offset_m, float(AVOID_PROXIMAL_OFFSET_M))
    ):
        print(f"Cached obstacle information shape mismatch, remeasuring: {OBSTACLE_INFO_PATH}")
        return None

    print(f"Loaded cached obstacle information from {OBSTACLE_INFO_PATH}")
    return obstacle_positions, obstacle_node_indices, obstacle_node_positions, obstacle_node_full_dofs


def load_cached_obstacle_positions() -> np.ndarray:
    if not OBSTACLE_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Cached obstacle information not found: {OBSTACLE_INFO_PATH}. "
            "Enable --capture-motion once to measure and save obstacle information."
        )
    try:
        data = np.load(OBSTACLE_INFO_PATH, allow_pickle=True)
        obstacle_positions = np.asarray(data["obstacle_positions"], dtype=float).reshape(-1, 3)
    except Exception as exc:
        raise ValueError(f"Failed to load cached obstacle positions from {OBSTACLE_INFO_PATH}: {exc}") from exc
    if obstacle_positions.size == 0 or not np.all(np.isfinite(obstacle_positions)):
        raise ValueError(f"Cached obstacle positions are empty or invalid: {OBSTACLE_INFO_PATH}")
    return obstacle_positions


def save_obstacle_information(
    obstacle_positions: np.ndarray,
    obstacle_node_indices_1based: np.ndarray,
    obstacle_node_positions: np.ndarray,
    obstacle_node_full_dofs_0based: np.ndarray,
    nodes_coord_constrained_count: int,
) -> None:
    OBSTACLE_INFO_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OBSTACLE_INFO_PATH,
        obstacle_positions=np.asarray(obstacle_positions, dtype=float).reshape(-1, 3),
        obstacle_node_indices_1based=np.asarray(obstacle_node_indices_1based, dtype=int).reshape(-1),
        obstacle_node_positions=np.asarray(obstacle_node_positions, dtype=float).reshape(-1, 3),
        obstacle_node_full_dofs_0based=np.asarray(obstacle_node_full_dofs_0based, dtype=int).reshape(-1),
        nodes_coord_constrained_count=int(nodes_coord_constrained_count),
        obstacle_rigid_ids=np.asarray(OBSTACLE_RIGID_IDS, dtype=int).reshape(-1),
        obstacle_radius=float(OBSTACLE_RADIUS),
        safe_distance_factor=float(SAFE_DISTANCE_FACTOR),
        avoid_constraint_target=str(AVOID_CONSTRAINT_TARGET),
        avoid_node_selection=str(AVOID_NODE_SELECTION),
        avoid_proximal_offset_m=float(AVOID_PROXIMAL_OFFSET_M),
    )
    print(f"Saved obstacle information to {OBSTACLE_INFO_PATH}")


def initialize_avoidance_reference(sdk, rom_data, use_cached_obstacles: bool = True) -> AvoidanceReference:
    sdk.ReceiveData(0)
    rigid_info = sdk.GetRigid()
    marker_info = sdk.GetMarker()

    base = _item_by_id(rigid_info, BASE_RIGID_ID)
    base_xyz = np.array([base.X, base.Y - BASE_Y_OFFSET, base.Z], dtype=float)
    base_q = np.array([base.qw, base.qx, base.qy, base.qz], dtype=float)
    motion = MotionReference(
        end_effector=np.zeros(3),
        mid_effector=np.zeros(3),
        base_xyz=base_xyz,
        base_to_global=quat2dcm(base_q),
    )

    motion.end_effector = _local_position(_xyz(_item_by_id(marker_info, END_MARKER_ID)), motion)
    motion.mid_effector = _local_position(_xyz(_item_by_id(rigid_info, MID_RIGID_ID)), motion)
    feedback_point = _local_position(_xyz(_item_by_id(rigid_info, FEEDBACK_RIGID_ID)), motion)
    motion.checkpoint = feedback_point

    cached_obstacle_info = load_obstacle_information(rom_data) if use_cached_obstacles else None
    if cached_obstacle_info is None:
        obstacle_positions = capture_obstacle_positions_in_model_frame(
            sdk,
            motion,
            obstacle_rigid_ids=OBSTACLE_RIGID_IDS,
            pos_end=POS_END,
        )
        motion.obstacles = obstacle_positions

        obstacle_node_indices, obstacle_node_positions, obstacle_node_full_dofs = obstacle_node_info(
            rom_data,
            obstacle_positions,
        )
        save_obstacle_information(
            obstacle_positions,
            obstacle_node_indices,
            obstacle_node_positions,
            obstacle_node_full_dofs,
            np.asarray(rom_data["nodes_coord"], dtype=float).shape[1],
        )
    else:
        obstacle_positions, obstacle_node_indices, obstacle_node_positions, obstacle_node_full_dofs = cached_obstacle_info
        motion.obstacles = obstacle_positions

    return AvoidanceReference(
        motion=motion,
        feedback_point=feedback_point,
        obstacle_positions=obstacle_positions,
        obstacle_node_indices_1based=obstacle_node_indices,
        obstacle_node_positions=obstacle_node_positions,
        obstacle_node_full_dofs_0based=obstacle_node_full_dofs,
    )


def node_index_from_dofs(dofs: np.ndarray) -> int:
    return int((np.asarray(dofs, dtype=int).reshape(-1)[0] + 2) // 3) - 1


def build_default_avoidance_reference(rom_data) -> AvoidanceReference:
    nodes_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    mid_index = node_index_from_dofs(np.asarray(rom_data["outdof_half"], dtype=int))
    mid_offset_mm = (nodes_xyz[mid_index] - POS_END) * 1000.0
    motion = MotionReference(
        end_effector=np.zeros(3),
        mid_effector=mid_offset_mm,
        checkpoint=mid_offset_mm,
        base_xyz=np.zeros(3),
        base_to_global=np.eye(3),
    )

    obstacle_positions = load_cached_obstacle_positions()
    motion.obstacles = obstacle_positions
    obstacle_node_indices, obstacle_node_positions, obstacle_node_full_dofs = obstacle_node_info(
        rom_data,
        obstacle_positions,
    )

    print(f"Loaded cached obstacle positions from {OBSTACLE_INFO_PATH}")
    print("Motion-capture measurement disabled; using model default reference.")
    return AvoidanceReference(
        motion=motion,
        feedback_point=motion.checkpoint,
        obstacle_positions=obstacle_positions,
        obstacle_node_indices_1based=obstacle_node_indices,
        obstacle_node_positions=obstacle_node_positions,
        obstacle_node_full_dofs_0based=obstacle_node_full_dofs,
    )


def cable_displacement_size(node_list) -> int:
    long_count = 3 * (node_list[0].shape[1] - 1) * 3
    short_count = 3 * (node_list[1].shape[1] - 1) * 3
    return long_count + short_count


def connect_sdk():
    sdk = load_lumo_sdk(BASE_DIR)
    sdk.Init()
    if not sdk.Connect(IP_ADDRESS):
        raise ConnectionError(f"Failed to connect LuMo SDK at {IP_ADDRESS}")
    print("LuMo SDK connected successfully")
    return sdk


def build_avoid_config(rom_data, avoid_ref: AvoidanceReference) -> SsmExperimentConfig:
    t_list = rom_data["t_list"]
    zd = rom_data["zd_proj"]
    node_list = rom_data["Node_List"]
    u_lb = np.full(6 * horizon, -50)
    u_ub = np.full(6 * horizon, 50)
    dist0 = cable_length_double_part(np.zeros(cable_displacement_size(node_list)), node_list)

    return SsmExperimentConfig(
        dt=CONTROL_PERIOD_SECONDS,
        tf=float(t_list[-1]),
        td=t_list,
        zd=zd,
        Np=horizon,
        Q=Q_WEIGHT,
        R=R_WEIGHT,
        u_lb=u_lb,
        u_ub=u_ub,
        Node_List=node_list,
        dist0=dist0,
        opt_W=36,
        solver_type=MPC_SOLVER_TYPE,
        qp_backend=QP_BACKEND,
        use_fast_single_step=USE_FAST_SINGLE_STEP_QP,
        control_smooth_weight=CONTROL_SMOOTH_WEIGHT,
        nmpc_solver=nmpc_multi_shooting_zoh_avoid_complex,
        pos_end=POS_END,
        pos_obstacle=avoid_ref.obstacle_positions,
        avoid_node_positions=avoid_ref.obstacle_node_positions,
        d_safe=SAFE_DISTANCE_FACTOR * OBSTACLE_RADIUS,
        # motor_safety_limit=35.0,
        motor_safety_limit=40.0,
    )


def cable_dofs_for_wp(node_list) -> np.ndarray:
    dofs = []
    for section_idx, last_node in [(0, 17), (1, 10)]:
        for cable_idx in range(3):
            for node_idx in range(1, last_node):
                dofs.extend(np.asarray(node_list[section_idx][cable_idx, node_idx]).reshape(-1)[3:6])
    return np.asarray(dofs, dtype=int)


def feedback_point_dofs(rom_data, feedback_position_mm: np.ndarray) -> np.ndarray:
    valid_node_indices, valid_nodes_coord = constrained_node_table(rom_data)
    feedback_position_m = (np.asarray(feedback_position_mm, dtype=float).reshape(3) / 1000.0) + POS_END
    distances = np.linalg.norm(valid_nodes_coord - feedback_position_m.reshape(3, 1), axis=0)
    full_node_idx = int(valid_node_indices[int(np.argmin(distances))])
    full_dofs = np.array([3 * full_node_idx, 3 * full_node_idx + 1, 3 * full_node_idx + 2], dtype=int)
    return full_dofs + 1


def build_ssm_maps(rom_data, avoid_ref: AvoidanceReference):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR

    n = 3 * np.asarray(rom_data["nodes_coord"], dtype=float).shape[1]
    outdof_endeff = np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)
    outdof_half = np.asarray(rom_data["outdof_half"], dtype=int).reshape(-1)
    outdof_feedback = feedback_point_dofs(rom_data, avoid_ref.feedback_point - avoid_ref.motion.end_effector)
    outdofs_for_wp = np.concatenate([outdof_endeff, outdof_half, outdof_endeff + n, outdof_half + n])
    outdofs_for_fb = np.concatenate([outdof_endeff, outdof_half, outdof_feedback])

    return (
        subset_mapping_rows(w0, outdofs_for_wp),
        subset_mapping_rows(w0, cable_dofs_for_wp(rom_data["Node_List"])),
        subset_mapping_rows(w0, outdofs_for_fb),
        subset_mapping_rows(w0, avoid_ref.obstacle_node_full_dofs_0based + 1),
    )


def build_initial_feedback_state(x_num: int, control_num: int, horizon: int) -> SsmFeedbackState:
    return SsmFeedbackState(
        t=0.0,
        k=0,
        pk=np.zeros(x_num),
        S0_opt=np.zeros((x_num, horizon)),
        ut_opt=np.zeros((control_num, horizon)),
    )


def load_recorded_control_input(control_input_path: str | Path) -> np.ndarray:
    path = Path(control_input_path)
    if not path.exists():
        raise FileNotFoundError(f"Recorded control input not found: {path}")

    control_input = np.asarray(np.load(path), dtype=float)
    if control_input.ndim != 2 or control_input.shape[1] != 6:
        raise ValueError(
            "Recorded control input must have shape (steps, 6), "
            f"got {control_input.shape} from {path}"
        )
    if not np.all(np.isfinite(control_input)):
        raise ValueError(f"Recorded control input contains NaN or Inf values: {path}")
    return control_input


@fixed_cpu_time(CONTROL_PERIOD_SECONDS)
def run_step(
    sdk,
    socket,
    controller,
    reference,
    step_index,
    y_expm,
    y_ssm,
    u_save,
    feedback=False,
    recorded_motor_positions=None,
    recorded_motor_factor=1.0,
    motion_capture_enabled=True,
):
    if motion_capture_enabled:
        measure_start = time.perf_counter()
        disp_end, disp_mid, disp_feedback = capture_motion(
            sdk,
            reference.motion,
            mid_rigid_id=MID_RIGID_ID,
            checkpoint_rigid_id=FEEDBACK_RIGID_ID,
            end_marker_id=END_MARKER_ID,
        )
        measure_time = time.perf_counter() - measure_start
        measurement = np.concatenate([disp_end, disp_mid, disp_feedback])
    else:
        measure_time = 0.0
        disp_end = np.full(3, np.nan)
        disp_mid = np.full(3, np.nan)
        disp_feedback = np.full(3, np.nan)
        measurement = None

    if recorded_motor_positions is None:
        mpc_start = time.perf_counter()
        motor_positions = controller.compute_next_command(measurement_mm=measurement, feedback=feedback)
        mpc_time = time.perf_counter() - mpc_start
        feedback_time = controller.last_feedback_time
        y_ssm[step_index, :6] = controller.yk_ssm
    else:
        motor_positions = np.asarray(recorded_motor_positions, dtype=float).reshape(6) * float(recorded_motor_factor)
        mpc_time = 0.0
        feedback_time = 0.0
        y_ssm[step_index, :6] = np.nan

    send_motor_command(socket, motor_positions.tolist(), solver_order=False)

    y_expm[step_index, :6] = np.concatenate([disp_end, disp_mid])
    y_expm[step_index, 6:] = disp_feedback
    u_save[step_index, :] = motor_positions
    return motor_positions, measure_time, mpc_time, feedback_time


def format_motor_positions(motor_positions) -> str:
    return "[" + ", ".join(f"{value:8.3f}" for value in motor_positions) + "]"


def print_timing_summary(timing_records):
    if not timing_records:
        print("\nTiming summary: no completed steps.")
        return
    total = np.array([item["total"] for item in timing_records], dtype=float)
    measure = np.array([item["measure"] for item in timing_records], dtype=float)
    feedback = np.array([item["feedback"] for item in timing_records], dtype=float)
    mpc = np.array([item["mpc"] for item in timing_records], dtype=float)
    print("\nTiming summary")
    print(f"  completed steps : {len(timing_records)}")
    print(f"  avg total       : {np.mean(total):.6f}s")
    print(f"  avg measure     : {np.mean(measure):.6f}s")
    print(f"  avg feedback    : {np.mean(feedback):.6f}s")
    print(f"  avg mpc         : {np.mean(mpc):.6f}s")
    print(f"  max mpc         : {np.max(mpc):.6f}s")


def resolve_trajectory_dir(trajectory_name: str) -> Path:
    key = trajectory_name.lower()
    if key not in TRAJECTORY_DIRS:
        choices = ", ".join(sorted(TRAJECTORY_DIRS))
        raise ValueError(f"Unknown trajectory '{trajectory_name}'. Choose one of: {choices}")
    return TRAJECTORY_DIRS[key]


def main(
    trajectory_name=TRAJECTORY_NAME,
    control_source=CONTROL_SOURCE,
    control_input_path=RECORDED_CONTROL_INPUT_PATH,
    motion_capture_enabled=CAPTURE_MOTION_ENABLED,
):
    original_stdout = sys.stdout
    logger = BufferedTeeLogger(original_stdout)
    sys.stdout = logger

    sdk = None
    context = None
    socket = None
    timing_records = []
    max_mpc_time = 0.0
    try:
        traj_dir = resolve_trajectory_dir(trajectory_name)
        print(f"Using SSM avoidance trajectory: {trajectory_name.lower()} ({traj_dir})")
        rom_data = load_rom_data(MODEL_INFO_DIR, traj_dir=traj_dir)
        x_num = np.asarray(get_field(rom_data["auData"], "lamd")).reshape(-1).size
        motion_capture_enabled = bool(motion_capture_enabled)
        if FEEDBACK_ENABLED and not motion_capture_enabled:
            raise ValueError("capture_motion must be enabled when FEEDBACK_ENABLED=True.")

        if motion_capture_enabled:
            sdk = connect_sdk()
            avoid_ref = initialize_avoidance_reference(sdk, rom_data)
        else:
            avoid_ref = build_default_avoidance_reference(rom_data)
        config = build_avoid_config(rom_data, avoid_ref)
        w0_for_out, w0_for_cables, w0_for_feedback, w0_for_avoid_nodes = build_ssm_maps(
            rom_data,
            avoid_ref,
        )

        print(f"Obstacle positions in model frame (m): {config.pos_obstacle}")
        for obstacle_idx, (node_index, node_position) in enumerate(
            zip(avoid_ref.obstacle_node_indices_1based, avoid_ref.obstacle_node_positions),
            start=1,
        ):
            print(f"  obstacle {obstacle_idx}: nearest node {node_index}, node position (m) {node_position}")
        print(f"Avoidance constraint target: {AVOID_CONSTRAINT_TARGET}")
        print(f"Safe distance (m): {config.d_safe:.6f}")

        controller = SsmMpcController(
            config=config,
            au_data=rom_data["auData"],
            w0_for_out=w0_for_out,
            w0_for_cables=w0_for_cables,
            w0_for_feedback=w0_for_feedback,
            w0_for_avoid_nodes=w0_for_avoid_nodes,
            feedback_state=build_initial_feedback_state(x_num, 6, config.Np),
            factor=MOTOR_FACTOR,
            feedback_enabled=FEEDBACK_ENABLED,
            save_feedback_each_step=SAVE_FEEDBACK_EACH_STEP,
        )

        context, socket = create_publisher(MOTOR_ENDPOINT)
        control_source = str(control_source).lower()
        if control_source not in ("dataset", "online"):
            raise ValueError("control_source must be 'dataset' or 'online'.")

        recorded_control_input = None
        if control_source == "dataset":
            recorded_control_input = load_recorded_control_input(control_input_path)
            print(
                "Using dataset control input: "
                f"{Path(control_input_path)} ({recorded_control_input.shape[0]} steps)"
            )
            print(f"Dataset control input motor factor: {MOTOR_FACTOR:.6g}")
        else:
            print("Using online MPC-computed control input.")
        print(f"Per-step capture_motion enabled: {motion_capture_enabled}")

        step_count = config.zd.shape[1] if np.isinf(RUN_STEPS) else min(int(RUN_STEPS), config.zd.shape[1])
        if recorded_control_input is not None:
            step_count = min(step_count, recorded_control_input.shape[0])
        y_expm = np.zeros((step_count, 9))
        y_ssm = np.zeros((step_count, 6))
        pk_save = np.full((step_count, x_num), np.nan)
        u_save = np.zeros((step_count, 6))

        for step_index in range(step_count):
            start = time.perf_counter()
            motor_positions, measure_time, mpc_time, feedback_time = run_step(
                sdk,
                socket,
                controller,
                avoid_ref,
                step_index,
                y_expm,
                y_ssm,
                u_save,
                feedback=FEEDBACK_ENABLED,
                recorded_motor_positions=(
                    None if recorded_control_input is None else recorded_control_input[step_index]
                ),
                recorded_motor_factor=MOTOR_FACTOR,
                motion_capture_enabled=motion_capture_enabled,
            )
            if recorded_control_input is None:
                pk_save[step_index, :] = controller.feedback_state.pk
            total_time = time.perf_counter() - start
            max_mpc_time = max(max_mpc_time, mpc_time)
            timing_records.append(
                {"total": total_time, "measure": measure_time, "feedback": feedback_time, "mpc": mpc_time}
            )
            print(
                f"Step {step_index + 1:04d}/{step_count:04d} | "
                f"total {total_time:8.4f}s | "
                f"measure {measure_time:8.4f}s | "
                f"feedback {feedback_time:8.4f}s | "
                f"mpc {mpc_time:8.4f}s | "
                f"mpc max {max_mpc_time:8.4f}s | "
                f"motor {format_motor_positions(motor_positions)}"
            )

        AVOID_COMPLEX_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        t_save = config.td[:step_count]
        np.save(AVOID_COMPLEX_RESULT_DIR / f"y_exprm_ssm_avoid{OUTPUT_SUFFIX}", y_expm)
        np.save(AVOID_COMPLEX_RESULT_DIR / f"y_ssm_avoid{OUTPUT_SUFFIX}", y_ssm)
        np.save(AVOID_COMPLEX_RESULT_DIR / f"pk_save_ssm_avoid{OUTPUT_SUFFIX}", pk_save)
        np.save(AVOID_COMPLEX_RESULT_DIR / f"ut_save_ssm_avoid{OUTPUT_SUFFIX}", u_save)
        np.save(AVOID_COMPLEX_RESULT_DIR / f"t_ssm_avoid{OUTPUT_SUFFIX}", t_save)
        sciIO.savemat(
            AVOID_COMPLEX_RESULT_DIR / f"ssm_avoid_python_results{OUTPUT_SUFFIX}.mat",
            {
                "y_exprm": y_expm,
                "y_ssm": y_ssm,
                "pk_save": pk_save,
                "ut_save": u_save,
                "t": t_save,
                "zzd": config.zd,
                "pos_end": POS_END,
                "pos_obstacle": config.pos_obstacle,
                "obstacle_node_indices_1based": avoid_ref.obstacle_node_indices_1based,
                "obstacle_node_positions": avoid_ref.obstacle_node_positions,
                "obstacle_node_full_dofs_0based": avoid_ref.obstacle_node_full_dofs_0based,
                "obstacle_radius": OBSTACLE_RADIUS,
                "d_safe": config.d_safe,
            },
        )
    finally:
        print_timing_summary(timing_records)
        if socket is not None and context is not None:
            close_publisher(context, socket)
        if sdk is not None:
            if hasattr(sdk, "Disconnect"):
                sdk.Disconnect()
            sdk.Close()
        sys.stdout = original_stdout
        log_path = logger.save(LOG_DIR)
        print(f"Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complex SSM obstacle-avoidance MPC.")
    parser.add_argument("--trajectory", choices=sorted(TRAJECTORY_DIRS), default=TRAJECTORY_NAME)
    parser.add_argument(
        "--control-source",
        choices=("dataset", "online"),
        default=CONTROL_SOURCE,
        help="Choose dataset replay input or online MPC-computed input.",
    )
    parser.add_argument(
        "--control-input",
        default=str(RECORDED_CONTROL_INPUT_PATH),
        help="Path to a recorded (steps, 6) motor control input .npy file when --control-source=dataset.",
    )
    motion_capture_group = parser.add_mutually_exclusive_group()
    motion_capture_group.add_argument(
        "--capture-motion",
        dest="motion_capture_enabled",
        action="store_true",
        default=CAPTURE_MOTION_ENABLED,
        help="Enable SDK motion capture for initialization and per-step measurements.",
    )
    motion_capture_group.add_argument(
        "--no-capture-motion",
        dest="motion_capture_enabled",
        action="store_false",
        help="Disable SDK motion capture; use cached obstacle information and save measured outputs as NaN.",
    )
    args = parser.parse_args()
    main(
        trajectory_name=args.trajectory,
        control_source=args.control_source,
        control_input_path=args.control_input,
        motion_capture_enabled=args.motion_capture_enabled,
    )
