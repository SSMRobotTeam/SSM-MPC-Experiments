#!/usr/bin/python
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import scipy.io as sciIO

from controller.nmpc import NmpcConfig
from controller.nmpc_pcs import nmpc_pcs_multi_shooting
from dynamics.pcs_model import PCSSection, Tendon, TwoSectionPCSRobot
from measure.motion_capture import capture_motion, initialize_motion_reference
from utils.sdk_utils import load_lumo_sdk
from utils.timing_utils import fixed_cpu_time
from zmq_message.zmq_control import clamp_motor_positions, close_publisher, create_publisher, send_motor_command


BASE_DIR = Path(__file__).resolve().parent
# TRAJ_DIR = BASE_DIR / "traj_data"
TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/spiral"
# TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/eight"
PLOT_DIR = BASE_DIR / "PlotFig"
RESULT_DIR = PLOT_DIR / "pcs_results"
MOTOR_ENDPOINT = "tcp://localhost:5555"

IP_ADDRESS = "127.0.0.1"
CONTROL_PERIOD_SECONDS = 0.075
MPC_HORIZON = 3

L1, L2 = 0.2225, 0.2175
D1, D2 = 0.016, 0.014
END_OFFSET = np.array([0.0, 0.0, L1 + L2])
PCS_UNITS_PER_SECTION = 2
PCS_UNIT_COUNT = 2 * PCS_UNITS_PER_SECTION
PCS_MODEL = "full"  # simplified: [kx, ky, tau, eps], full: [kx, ky, tau, eps, gx, gy]
MOTOR_FACTOR = 1.01

PCS_MAX_COMMAND_MM = 40.0
PCS_MAX_COMMAND_STEP_MM = 10

PCS_MPC_Q = np.diag([5e3, 5e3, 5e3])
PCS_MPC_R = 1e-5
PCS_MPC_R_SMOOTH = 1

PCS_MPC_R_STRAIN = 1
PCS_MPC_R_BEND = 0.05
PCS_MPC_R_TWIST = 10
PCS_MPC_R_AXIAL = 10
PCS_MPC_R_SHEAR = 50
PCS_MPC_MAX_NFEV = 1e2
PCS_MPC_SOLVER = "qp"  # "ipopt" for nonlinear MPC, "qp" for linearized CasADi QP
PCS_MPC_QP_BACKEND = "qpoases"


def pcs_strain_size() -> int:
    if PCS_MODEL == "simplified":
        return 4
    if PCS_MODEL == "full":
        return 6
    raise ValueError(f"Unknown PCS_MODEL: {PCS_MODEL}")


def zero_pcs_strains(robot: TwoSectionPCSRobot) -> np.ndarray:
    return np.zeros((PCS_UNIT_COUNT, robot.strain_size), dtype=float)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def create_log_path() -> Path:
    log_dir = PLOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"pcs_log_{timestamp}.txt"


def make_robot() -> TwoSectionPCSRobot:
    """Build a two-section PCS model discretized into four PCS units."""

    long_angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    short_angles = np.array([np.pi, np.pi / 3.0, 5.0 * np.pi / 3.0])

    section_1_units = [
        PCSSection.with_uniform_tendons(length=L1 / PCS_UNITS_PER_SECTION, radius=D1, angles=long_angles)
        for _ in range(PCS_UNITS_PER_SECTION)
    ]
    section_2_units = [
        PCSSection.with_uniform_tendons(length=L2 / PCS_UNITS_PER_SECTION, radius=D2, angles=long_angles)
        for _ in range(PCS_UNITS_PER_SECTION)
    ]
    short_tendons = tuple(Tendon(angle=float(angle), radius=D1) for angle in short_angles)
    return TwoSectionPCSRobot(
        [*section_1_units, *section_2_units],
        short_tendons=short_tendons,
        short_section_count=PCS_UNITS_PER_SECTION,
        strain_size=pcs_strain_size(),
    )


def load_trajectory() -> tuple[np.ndarray, np.ndarray]:
    zd = sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"]
    if zd.ndim != 2 or zd.shape[0] != 3:
        raise ValueError(f"Expected zd_proj to have shape (3, N), got {zd.shape}")

    t_list = sciIO.loadmat(str(TRAJ_DIR / "t_list.mat"))["t_list"].squeeze()
    return zd + END_OFFSET.reshape(3, 1), t_list


def build_pcs_mpc_config() -> NmpcConfig:
    return NmpcConfig(
        Q=PCS_MPC_Q,
        R=PCS_MPC_R,
        Np=MPC_HORIZON,
        u_lb=np.full(6, -PCS_MAX_COMMAND_MM / 1000.0),
        u_ub=np.full(6, PCS_MAX_COMMAND_MM / 1000.0),
        solver_type=PCS_MPC_SOLVER,
        qp_backend=PCS_MPC_QP_BACKEND,
        use_fast_single_step=False,
    )


def reference_window(desired_positions: np.ndarray, step_index: int, horizon: int) -> np.ndarray:
    last_index = desired_positions.shape[1] - 1
    indices = np.clip(np.arange(step_index, step_index + horizon), 0, last_index)
    return desired_positions[:, indices]


def connect_sdk():
    sdk = load_lumo_sdk(BASE_DIR)
    sdk.Init()
    if sdk.Connect(IP_ADDRESS):
        print("LuMo SDK connected successfully")
    else:
        raise ConnectionError(f"Failed to connect LuMo SDK at {IP_ADDRESS}")
    return sdk


def solve_pcs_position(
    robot: TwoSectionPCSRobot,
    desired_positions: np.ndarray,
    q0: np.ndarray | None,
    mpc_config: NmpcConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    q_prev = (
        zero_pcs_strains(robot)
        if q0 is None
        else np.asarray(q0, dtype=float).reshape(PCS_UNIT_COUNT, robot.strain_size).copy()
    )

    mpc_start_time = time.perf_counter()
    result = nmpc_pcs_multi_shooting(
        robot,
        q_prev,
        desired_positions,
        mpc_config,
        zero_strains=zero_pcs_strains(robot),
        max_command_step_m=PCS_MAX_COMMAND_STEP_MM / 1000.0,
        R_strain=PCS_MPC_R_STRAIN,
        R_smooth=PCS_MPC_R_SMOOTH,
        R_bend=PCS_MPC_R_BEND,
        R_twist=PCS_MPC_R_TWIST,
        R_axial=PCS_MPC_R_AXIAL,
        R_shear=PCS_MPC_R_SHEAR,
        max_nfev=PCS_MPC_MAX_NFEV,
    )
    mpc_elapsed_time = time.perf_counter() - mpc_start_time
    q = limit_pcs_command_amplitude(robot, result.strains, q_prev)
    zero_q = zero_pcs_strains(robot)
    tendon_delta = robot.tendon_lengths(q) - robot.tendon_lengths(zero_q)
    post_limit_model_error = np.linalg.norm(tendon_delta - result.tendon_delta, ord=np.inf)
    print(
        f"PCS NMPC cost = {result.cost:.6e}, nfev = {result.nfev}, "
        f"success = {result.success}, mpc time = {mpc_elapsed_time:.6f}s, "
        f"model error = {result.model_error:.6e}m, post-limit delta = {post_limit_model_error:.6e}m"
    )
    return tendon_delta, q, mpc_elapsed_time


def limit_pcs_command_amplitude(
    robot: TwoSectionPCSRobot,
    q_candidate: np.ndarray,
    q_prev: np.ndarray,
) -> np.ndarray:
    q_limited = q_candidate.copy()
    zero_q = zero_pcs_strains(robot)
    zero_lengths = robot.tendon_lengths(zero_q)

    prev_command = robot.tendon_lengths(q_prev) - zero_lengths
    candidate_command = robot.tendon_lengths(q_limited) - zero_lengths
    command_step = candidate_command - prev_command
    max_step = np.max(np.abs(command_step))
    max_step_m = PCS_MAX_COMMAND_STEP_MM / 1000.0
    if max_step > max_step_m:
        scale = max_step_m / max_step
        q_limited = q_prev + scale * (q_limited - q_prev)
        print(f"PCS command step limited: scale = {scale:.3f}")

    candidate_command = robot.tendon_lengths(q_limited) - zero_lengths
    max_command = np.max(np.abs(candidate_command))
    max_command_m = PCS_MAX_COMMAND_MM / 1000.0
    if max_command > max_command_m:
        scale = max_command_m / max_command
        q_limited = scale * q_limited
        print(f"PCS command amplitude limited: scale = {scale:.3f}")

    return q_limited


def pcs_section_end_positions(robot: TwoSectionPCSRobot, strains: np.ndarray) -> np.ndarray:
    transforms = robot.forward_kinematics(strains)
    section_1_position = transforms[PCS_UNITS_PER_SECTION - 1][:3, 3]
    section_2_position = transforms[-1][:3, 3]
    return np.hstack([section_1_position, section_2_position])


@fixed_cpu_time(target_cpu_seconds=CONTROL_PERIOD_SECONDS)
def run_control_step(
    sdk,
    socket,
    robot: TwoSectionPCSRobot,
    mpc_config: NmpcConfig,
    step_index: int,
    desired_positions: np.ndarray,
    q0: np.ndarray | None,
    reference,
    y_expm: np.ndarray,
    q_save: np.ndarray,
    u_save: np.ndarray,
    pcs_position_save: np.ndarray,
) -> tuple[np.ndarray, float]:
    ref_positions = reference_window(desired_positions, step_index, mpc_config.Np)
    u_sol, q_sol, mpc_elapsed_time = solve_pcs_position(robot, ref_positions, q0, mpc_config)

    motor_positions_mm = clamp_motor_positions(u_sol * 1000.0 * MOTOR_FACTOR)
    print(f"control action = {motor_positions_mm}")
    send_motor_command(socket, motor_positions_mm, solver_order=True)

    disp_end, disp_mid, _ = capture_motion(sdk, reference)
    y_expm[step_index, :] = np.hstack([disp_end, disp_mid])
    q_save[step_index, :] = q_sol.reshape(-1)
    u_save[step_index, :] = np.asarray(motor_positions_mm, dtype=float)
    pcs_position_save[step_index, :] = pcs_section_end_positions(robot, q_sol)
    return q_sol, mpc_elapsed_time


def save_results(
    y_expm: np.ndarray,
    t_list: np.ndarray,
    q_save: np.ndarray,
    u_save: np.ndarray,
    pcs_position_save: np.ndarray,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / "y_expm_pcs", y_expm)
    np.save(RESULT_DIR / "t_expm_pcs", t_list)
    np.save(RESULT_DIR / "q_save_pcs", q_save)
    np.save(RESULT_DIR / "u_save_pcs", u_save)
    np.save(RESULT_DIR / "pcs_section_positions_pcs", pcs_position_save)


def run_experiment() -> None:
    sdk = connect_sdk()
    context = None
    socket = None

    try:
        robot = make_robot()
        mpc_config = build_pcs_mpc_config()
        print(f"PCS model = {PCS_MODEL}, strain size per unit = {robot.strain_size}")
        print(
            f"PCS NMPC solver = {mpc_config.solver_type}, horizon = {mpc_config.Np}, "
            f"fast single step = {mpc_config.use_fast_single_step}, "
            f"control period = {CONTROL_PERIOD_SECONDS:.3f}s"
        )
        desired_positions, t_list = load_trajectory()
        step_count = desired_positions.shape[1]

        y_expm = np.zeros((step_count, 6))
        q_save = np.zeros((step_count, PCS_UNIT_COUNT * robot.strain_size))
        u_save = np.zeros((step_count, 6))
        pcs_position_save = np.zeros((step_count, 6))

        reference = initialize_motion_reference(sdk)
        context, socket = create_publisher(MOTOR_ENDPOINT)
        print("ZeroMQ publisher started on /controller/setActuatorDate")

        q0 = None
        max_mpc_elapsed_time = 0.0
        for step_index in range(step_count):
            start_time = time.perf_counter()
            q0, mpc_elapsed_time = run_control_step(
                sdk=sdk,
                socket=socket,
                robot=robot,
                mpc_config=mpc_config,
                step_index=step_index,
                desired_positions=desired_positions,
                q0=q0,
                reference=reference,
                y_expm=y_expm,
                q_save=q_save,
                u_save=u_save,
                pcs_position_save=pcs_position_save,
            )
            max_mpc_elapsed_time = max(max_mpc_elapsed_time, mpc_elapsed_time)
            elapsed_time = time.perf_counter() - start_time
            print(
                f"Step {step_index}: running time = {elapsed_time:.6f}s, "
                f"mpc time = {mpc_elapsed_time:.6f}s, "
                f"max mpc step time = {max_mpc_elapsed_time:.6f}s\n"
            )

        save_results(y_expm, t_list[:step_count], q_save, u_save, pcs_position_save)
        print(y_expm)

    finally:
        if socket is not None and context is not None:
            close_publisher(context, socket)
        if sdk is not None:
            sdk.Close()


def main() -> None:
    log_path = create_log_path()
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = Tee(sys.stdout, log_file)
        stderr_tee = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"PCS log file: {log_path}")
            run_experiment()


if __name__ == "__main__":
    main()
