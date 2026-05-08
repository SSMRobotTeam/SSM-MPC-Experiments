#!/usr/bin/python
# C:\Users\beau\AppData\Local\Programs\Python\Python311\python.exe -m pip install
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import numpy as np
import scipy.io as sciIO

from controller.cable_constraints import cable_length_double_part
from controller.ssm_controller import SsmExperimentConfig, SsmFeedbackState, SsmMpcController
from dynamics.SSM.matlab_data import EIGHT_TRAJ_DIR, ROM_DIR, clone_struct_array, get_field, load_rom_data, subset_mapping_rows
from measure.motion_capture import capture_motion, initialize_motion_reference
from utils.sdk_utils import load_lumo_sdk
from utils.timing_utils import fixed_cpu_time
from zmq_message.zmq_control import close_publisher, create_publisher, send_motor_command


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "PlotFig"
LOG_DIR = PLOT_DIR / "logs"
IP_ADDRESS = "127.0.0.1"
MOTOR_ENDPOINT = "tcp://localhost:5555"
CONTROL_PERIOD_SECONDS = 0.075
SSM_FACTOR = 1.475
MOTOR_FACTOR = 1.225
horizon = 1
RUN_STEPS = np.inf # determined by desired trajectory
FEEDBACK_ENABLED = False
MPC_SOLVER_TYPE = "ipopt"  # choose one: "ipopt" or "qp"
QP_BACKEND = "qpoases"  # used only when MPC_SOLVER_TYPE = "qp"
USE_FAST_SINGLE_STEP_QP = True
SAVE_FEEDBACK_EACH_STEP = False
END_MARKER_ID = 100023
CHECKPOINT_RIGID_ID = 2
POS_END = np.array([0.0, 0.0, 0.4400])


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
        log_path = log_dir / f"SSM_log_{beijing_now:%Y%m%d_%H%M}.txt"
        if log_path.exists():
            index = 2
            while True:
                candidate = log_dir / f"SSM_log_{beijing_now:%Y%m%d_%H%M}_{index:02d}.txt"
                if not candidate.exists():
                    log_path = candidate
                    break
                index += 1
        log_path.write_text("".join(self.messages), encoding="utf-8")
        return log_path


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


def build_default_config(rom_data) -> SsmExperimentConfig:
    t_list = rom_data["t_list"]
    zd = rom_data["zd_proj"]
    dt = CONTROL_PERIOD_SECONDS
    node_list = rom_data["Node_List"]

    # These defaults mirror the MATLAB controller shape; tune weights in this function if needed.
    q_weight = 1 * np.diag([1e5, 1e5, 1e5, 1e3, 1e3, 1e3])
    r_weight = 1e-2
    u_lb = np.full(6 * horizon, -50)
    u_ub = np.full(6 * horizon,  50)
    dist0 = cable_length_double_part(np.zeros(cable_displacement_size(node_list)), node_list)

    return SsmExperimentConfig(
        dt=dt,
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
        opt_W=1.25,
        solver_type=MPC_SOLVER_TYPE,
        qp_backend=QP_BACKEND,
        use_fast_single_step=USE_FAST_SINGLE_STEP_QP,
    )


def cable_dofs_for_wp(node_list) -> np.ndarray:
    dofs = []
    for section_idx, last_node in [(0, 17), (1, 10)]:
        for cable_idx in range(3):
            for node_idx in range(1, last_node):
                dofs.extend(np.asarray(node_list[section_idx][cable_idx, node_idx]).reshape(-1)[3:6])
    return np.asarray(dofs, dtype=int)


def feedback_point_node_info(rom_data, reference) -> dict[str, np.ndarray | int | float]:
    feedback_position_m = ((reference.checkpoint - reference.end_effector) / 1000.0) + POS_END
    nodes_coord = np.asarray(rom_data["nodes_coord"], dtype=float)
    distances = np.linalg.norm(nodes_coord - feedback_position_m.reshape(3, 1), axis=0)
    node_index_1based = int(np.argmin(distances)) + 1
    return {
        "position_m": feedback_position_m,
        "node_index_1based": node_index_1based,
        "distance_m": float(distances[node_index_1based - 1]),
        "dofs": np.array([3 * node_index_1based - 2, 3 * node_index_1based - 1, 3 * node_index_1based]),
    }


def feedback_point_dofs(rom_data, reference) -> np.ndarray:
    return np.asarray(feedback_point_node_info(rom_data, reference)["dofs"], dtype=int)


def _format_array(values) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in np.asarray(values).reshape(-1)) + "]"


def print_measurement_node_info(rom_data, reference) -> None:
    outdof_endeff = np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)
    outdof_half = np.asarray(rom_data["outdof_half"], dtype=int).reshape(-1)
    feedback_info = feedback_point_node_info(rom_data, reference)
    outdof_feedback_pt = np.asarray(feedback_info["dofs"], dtype=int).reshape(-1)

    print("\nMotion-capture measurement node info")
    print(f"  endeff      initial Ref position (mm): {_format_array(reference.end_effector)}")
    print(f"  endeff      3-axis dofs              : {_format_array(outdof_endeff)}")
    print(f"  half        initial Ref position (mm): {_format_array(reference.mid_effector)}")
    print(f"  half        3-axis dofs              : {_format_array(outdof_half)}")
    print(f"  feedback_pt initial Ref position (mm): {_format_array(reference.checkpoint)}")
    print(f"  feedback_pt model position (m)       : {_format_array(feedback_info['position_m'])}")
    print(f"  feedback_pt nearest node             : {feedback_info['node_index_1based']}")
    print(f"  feedback_pt node distance (m)        : {feedback_info['distance_m']:.6g}")
    print(f"  feedback_pt 3-axis dofs              : {_format_array(outdof_feedback_pt)}")


def build_ssm_maps(rom_data, reference):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR

    n = 5574
    outdof_endeff = np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)
    outdof_half = np.asarray(rom_data["outdof_half"], dtype=int).reshape(-1)
    outdof_feedback_pt = feedback_point_dofs(rom_data, reference)
    outdofs_for_wp = np.concatenate([outdof_endeff, outdof_half, outdof_endeff + n, outdof_half + n])
    cables_dofs = cable_dofs_for_wp(rom_data["Node_List"])

    outdofs_for_fb = np.concatenate([outdof_endeff, outdof_half, outdof_feedback_pt])
    return (
        subset_mapping_rows(w0, outdofs_for_wp),
        subset_mapping_rows(w0, cables_dofs),
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
def run_step(sdk, socket, controller, reference, step_index, y_expm, u_save, end_marker_id, checkpoint_rigid_id, feedback=False):
    measure_start = time.perf_counter()
    disp_end, disp_mid, disp_checkpoint = capture_motion(
        sdk,
        reference,
        checkpoint_rigid_id=checkpoint_rigid_id,
        end_marker_id=end_marker_id,
    )
    measure_time = time.perf_counter() - measure_start

    measurement = np.concatenate([disp_end, disp_mid, disp_checkpoint]) # 测量
    mpc_start = time.perf_counter()
    motor_positions = controller.compute_next_command(measurement_mm=measurement, feedback=feedback)
    mpc_time = time.perf_counter() - mpc_start
    feedback_time = controller.last_feedback_time

    send_motor_command(socket, motor_positions.tolist(), solver_order=False)

    y_expm[step_index, :6] = np.concatenate([disp_end, disp_mid])
    y_expm[step_index, 6:] = measurement[6:]
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


def main(end_marker_id=END_MARKER_ID):
    original_stdout = sys.stdout
    logger = BufferedTeeLogger(original_stdout)
    sys.stdout = logger

    sdk = None
    context = None
    socket = None
    timing_records = []
    max_mpc_time = 0.0
    try:
        rom_data = load_rom_data(ROM_DIR, traj_dir=EIGHT_TRAJ_DIR)
        config = build_default_config(rom_data)
        x_num = np.asarray(get_field(rom_data["auData"], "lamd")).reshape(-1).size

        sdk = connect_sdk()
        reference = initialize_motion_reference(
            sdk,
            checkpoint_rigid_id=CHECKPOINT_RIGID_ID,
            end_marker_id=end_marker_id,
        )
        w0_for_out, w0_for_cables, w0_for_feedback = build_ssm_maps(rom_data, reference)
        print_measurement_node_info(rom_data, reference)

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
        u_save = np.zeros((step_count, 6))

        for step_index in range(step_count):
            start = time.perf_counter()
            motor_positions, measure_time, mpc_time, feedback_time = run_step(
                sdk,
                socket,
                controller,
                reference,
                step_index,
                y_expm,
                u_save,
                end_marker_id,
                CHECKPOINT_RIGID_ID,
                feedback=FEEDBACK_ENABLED,
            )
            total_time = time.perf_counter() - start
            max_mpc_time = max(max_mpc_time, mpc_time)
            timing_records.append(
                {
                    "total": total_time,
                    "measure": measure_time,
                    "feedback": feedback_time,
                    "mpc": mpc_time,
                }
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
        np.save(PLOT_DIR / "y_exprm_ssm", y_expm)
        np.save(PLOT_DIR / "ut_save_ssm", u_save)
        np.save(PLOT_DIR / "t_ssm", config.td[:step_count])
        sciIO.savemat(PLOT_DIR / "ssm_python_results.mat", {"y_exprm": y_expm, "ut_save": u_save, "t": config.td[:step_count]})
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
