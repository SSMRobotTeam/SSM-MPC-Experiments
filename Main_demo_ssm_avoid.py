#!/usr/bin/python
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import numpy as np
import scipy.io as sciIO

from controller.cable_constraints import cable_length_double_part
from controller.nmpc import nmpc_multi_shooting_zoh_avoid_soft
from controller.ssm_controller import SsmExperimentConfig, SsmFeedbackState, SsmMpcController
from dynamics.SSM.matlab_data import SPIRAL_TRAJ_DIR, clone_struct_array, get_field, load_rom_data, subset_mapping_rows
from measure.motion_capture import MotionReference, capture_motion
from utils.quat_rotR import quat2dcm
from utils.sdk_utils import load_lumo_sdk
from utils.timing_utils import fixed_cpu_time
from zmq_message.zmq_control import close_publisher, create_publisher, send_motor_command


BASE_DIR = Path(__file__).resolve().parent
MODEL_INFO_DIR = BASE_DIR / "dynamics" / "modelinfom"
PLOT_DIR = BASE_DIR / "PlotFig"
LOG_DIR = PLOT_DIR / "logs"
IP_ADDRESS = "127.0.0.1"
MOTOR_ENDPOINT = "tcp://localhost:5555"
CONTROL_PERIOD_SECONDS = 0.075
SSM_FACTOR = 1.5
MOTOR_FACTOR = 1.3
horizon = 1
RUN_STEPS = np.inf
FEEDBACK_ENABLED = False
MPC_SOLVER_TYPE = "ipopt"  # choose one: "ipopt" or "qp"
QP_BACKEND = "qpoases"  # used only when MPC_SOLVER_TYPE = "qp"
USE_FAST_SINGLE_STEP_QP = False
CONTROL_SMOOTH_WEIGHT = 1e2
SAVE_FEEDBACK_EACH_STEP = False
END_MARKER_ID = 100027
BASE_RIGID_ID = 0
MID_RIGID_ID = 1
FEEDBACK_RIGID_ID = 2
OBSTACLE_RIGID_ID = 3
BASE_Y_OFFSET = -12.0
POS_END = np.array([0.0, 0.0, 0.4400])
OBSTACLE_RADIUS = 0.03
SAFE_DISTANCE_FACTOR = 2.0


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
        log_path = log_dir / f"SSM_avoid_log_{beijing_now:%Y%m%d_%H%M}.txt"
        if log_path.exists():
            index = 2
            while True:
                candidate = log_dir / f"SSM_avoid_log_{beijing_now:%Y%m%d_%H%M}_{index:02d}.txt"
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
    obstacle_displacement_from_end: np.ndarray


def _xyz(obj) -> np.ndarray:
    return np.array([obj.X, obj.Y, obj.Z], dtype=float)


def _item_by_id(items, item_id):
    try:
        return items[item_id]
    except Exception:
        pass
    for item in items:
        if getattr(item, "markerID", None) == item_id or getattr(item, "id", None) == item_id:
            return item
    raise KeyError(f"Could not find motion-capture item id {item_id}")


def _local_position(point_xyz: np.ndarray, reference: MotionReference) -> np.ndarray:
    return (reference.base_to_global @ (point_xyz - reference.base_xyz).T).T


def initialize_avoidance_reference(sdk) -> AvoidanceReference:
    sdk.ReceiveData(0)
    rigid_info = sdk.GetRigid()
    marker_info = sdk.GetMarker()

    base = rigid_info[BASE_RIGID_ID]
    base_xyz = np.array([base.X, base.Y - BASE_Y_OFFSET, base.Z], dtype=float)
    base_q = np.array([base.qw, base.qx, base.qy, base.qz], dtype=float)
    motion = MotionReference(
        end_effector=np.zeros(3),
        mid_effector=np.zeros(3),
        base_xyz=base_xyz,
        base_to_global=quat2dcm(base_q),
    )

    motion.end_effector = _local_position(_xyz(_item_by_id(marker_info, END_MARKER_ID)), motion)
    motion.mid_effector = _local_position(_xyz(rigid_info[MID_RIGID_ID]), motion)
    feedback_point = _local_position(_xyz(rigid_info[FEEDBACK_RIGID_ID]), motion)
    motion.checkpoint = feedback_point
    obstacle_point = _local_position(_xyz(rigid_info[OBSTACLE_RIGID_ID]), motion)

    return AvoidanceReference(
        motion=motion,
        feedback_point=feedback_point,
        obstacle_displacement_from_end=(obstacle_point - motion.end_effector) / 1000.0,
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
    q_weight = np.diag([1e5, 1e5, 1e5, 1e3, 1e3, 1e3])
    r_weight = 1e-2
    u_lb = np.full(6 * horizon, -50)
    u_ub = np.full(6 * horizon, 50)
    dist0 = cable_length_double_part(np.zeros(cable_displacement_size(node_list)), node_list)
    pos_obstacle = avoid_ref.obstacle_displacement_from_end + POS_END

    return SsmExperimentConfig(
        dt=CONTROL_PERIOD_SECONDS,
        tf=float(t_list[-1]),
        td=t_list,
        zd=zd,
        Np=horizon,
        Q=q_weight,
        R=r_weight,
        u_lb=u_lb,
        u_ub=u_ub,
        Node_List=node_list,
        dist0=dist0,
        opt_W=36,
        solver_type=MPC_SOLVER_TYPE,
        qp_backend=QP_BACKEND,
        use_fast_single_step=USE_FAST_SINGLE_STEP_QP,
        control_smooth_weight=CONTROL_SMOOTH_WEIGHT,
        nmpc_solver=nmpc_multi_shooting_zoh_avoid_soft,
        pos_end=POS_END,
        pos_obstacle=pos_obstacle,
        d_safe=(SAFE_DISTANCE_FACTOR * OBSTACLE_RADIUS) ** 2,
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
    nodes_coord = np.asarray(rom_data["nodes_coord"], dtype=float)
    feedback_position_m = (np.asarray(feedback_position_mm, dtype=float).reshape(3) / 1000.0) + POS_END
    distances = np.linalg.norm(nodes_coord - feedback_position_m.reshape(3, 1), axis=0)
    node_index_1based = int(np.argmin(distances)) + 1
    return np.array([3 * node_index_1based - 2, 3 * node_index_1based - 1, 3 * node_index_1based])


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
    )


def build_initial_feedback_state(x_num: int, control_num: int, horizon: int) -> SsmFeedbackState:
    return SsmFeedbackState(
        t=0.0,
        k=0,
        pk=np.zeros(x_num),
        S0_opt=np.zeros((x_num, horizon)),
        ut_opt=np.zeros((control_num, horizon)),
    )


@fixed_cpu_time(CONTROL_PERIOD_SECONDS)
def run_step(sdk, socket, controller, reference, step_index, y_expm, y_ssm, u_save, feedback=False):
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
    mpc_start = time.perf_counter()
    motor_positions = controller.compute_next_command(measurement_mm=measurement, feedback=feedback)
    mpc_time = time.perf_counter() - mpc_start
    feedback_time = controller.last_feedback_time

    send_motor_command(socket, motor_positions.tolist(), solver_order=False)

    y_expm[step_index, :6] = np.concatenate([disp_end, disp_mid])
    y_expm[step_index, 6:] = measurement[6:]
    y_ssm[step_index, :6] = controller.yk_ssm
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


def main():
    original_stdout = sys.stdout
    logger = BufferedTeeLogger(original_stdout)
    sys.stdout = logger

    sdk = None
    context = None
    socket = None
    timing_records = []
    max_mpc_time = 0.0
    try:
        rom_data = load_rom_data(MODEL_INFO_DIR, traj_dir=SPIRAL_TRAJ_DIR)
        x_num = np.asarray(get_field(rom_data["auData"], "lamd")).reshape(-1).size

        sdk = connect_sdk()
        avoid_ref = initialize_avoidance_reference(sdk)
        config = build_avoid_config(rom_data, avoid_ref)
        w0_for_out, w0_for_cables, w0_for_feedback = build_ssm_maps(rom_data, avoid_ref)

        print(f"Obstacle displacement from end (m): {avoid_ref.obstacle_displacement_from_end}")
        print(f"Obstacle position in model frame (m): {config.pos_obstacle}")
        print(f"Safe distance squared (m^2): {config.d_safe:.6f}")

        controller = SsmMpcController(
            config=config,
            au_data=rom_data["auData"],
            w0_for_out=w0_for_out,
            w0_for_cables=w0_for_cables,
            w0_for_feedback=w0_for_feedback,
            feedback_state=build_initial_feedback_state(x_num, 6, config.Np),
            factor=MOTOR_FACTOR,
            feedback_enabled=FEEDBACK_ENABLED,
            save_feedback_each_step=SAVE_FEEDBACK_EACH_STEP,
        )

        context, socket = create_publisher(MOTOR_ENDPOINT)
        step_count = config.zd.shape[1] if np.isinf(RUN_STEPS) else min(int(RUN_STEPS), config.zd.shape[1])
        y_expm = np.zeros((step_count, 9))
        y_ssm = np.zeros((step_count, 6))
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
            )
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

        PLOT_DIR.mkdir(exist_ok=True)
        t_save = config.td[:step_count]
        np.save(PLOT_DIR / "y_exprm_ssm_avoid", y_expm)
        np.save(PLOT_DIR / "y_ssm_avoid", y_ssm)
        np.save(PLOT_DIR / "ut_save_ssm_avoid", u_save)
        np.save(PLOT_DIR / "t_ssm_avoid", t_save)
        sciIO.savemat(
            PLOT_DIR / "ssm_avoid_python_results.mat",
            {
                "y_exprm": y_expm,
                "y_ssm": y_ssm,
                "ut_save": u_save,
                "t": t_save,
                "zzd": config.zd,
                "pos_end": POS_END,
                "pos_obstacle": config.pos_obstacle,
                "obstacle_radius": OBSTACLE_RADIUS,
                "d_safe": config.d_safe,
            },
        )
    finally:
        print_timing_summary(timing_records)
        if socket is not None and context is not None:
            close_publisher(context, socket)
        if sdk is not None:
            sdk.Close()
        sys.stdout = original_stdout
        log_path = logger.save(LOG_DIR)
        print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
