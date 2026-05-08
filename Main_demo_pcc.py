#!/usr/bin/python
from pathlib import Path
import time

import numpy as np
import scipy.io as sciIO

from dynamics.pcc_inverseKinematic import _section_rotation, _section_tip, ik_position
from measure.motion_capture import capture_motion, initialize_motion_reference
from utils.sdk_utils import load_lumo_sdk
from utils.timing_utils import fixed_cpu_time
from zmq_message.zmq_control import close_publisher, create_publisher, send_motor_command


BASE_DIR = Path(__file__).resolve().parent
# TRAJ_DIR = BASE_DIR / "traj_data"
# TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/spiral"
TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/eight"
PLOT_DIR = BASE_DIR / "PlotFig"
MOTOR_ENDPOINT = "tcp://localhost:5555"

IP_ADDRESS = "127.0.0.1"
CONTROL_PERIOD_SECONDS = 0.075

L1, L2 = 0.2225, 0.2175
D1, D2 = 0.0165, 0.0145
END_OFFSET = np.array([0.0, 0.0, L1 + L2])


def load_trajectory() -> tuple[np.ndarray, np.ndarray]:
    zd = sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"]
    if zd.ndim != 2 or zd.shape[0] != 3:
        raise ValueError(f"Expected zd_proj to have shape (3, N), got {zd.shape}")

    t_list = sciIO.loadmat(str(TRAJ_DIR / "t_list.mat"))["t_list"].squeeze()
    return zd + END_OFFSET.reshape(3, 1), t_list


def connect_sdk():
    sdk = load_lumo_sdk(BASE_DIR)
    sdk.Init()
    if sdk.Connect(IP_ADDRESS):
        print("LuMo SDK connected successfully")
    else:
        raise ConnectionError(f"Failed to connect LuMo SDK at {IP_ADDRESS}")
    return sdk


def pcc_section_end_positions(q: np.ndarray) -> np.ndarray:
    theta1, phi1, theta2, phi2 = np.asarray(q, dtype=float)
    section_1_position = _section_tip(theta1, phi1, L1)
    section_2_position = section_1_position + _section_rotation(theta1, phi1) @ _section_tip(theta2, phi2, L2)
    return np.hstack([section_1_position, section_2_position])


@fixed_cpu_time(target_cpu_seconds=CONTROL_PERIOD_SECONDS)
def run_control_step(
    sdk,
    socket,
    step_index: int,
    desired_position: np.ndarray,
    q0: np.ndarray | None,
    reference,
    y_expm: np.ndarray,
    q_save: np.ndarray,
    u_save: np.ndarray,
    pcc_position_save: np.ndarray,
) -> np.ndarray:
    u_sol, q_sol, _ = ik_position(desired_position, L1, L2, D1, D2, q0=q0)

    motor_positions_mm = (u_sol * 1000.0).tolist()
    print(f"control action = {motor_positions_mm}")
    send_motor_command(socket, motor_positions_mm, solver_order=True)

    disp_end, disp_mid, _ = capture_motion(sdk, reference)
    y_expm[step_index, :] = np.hstack([disp_end, disp_mid])
    q_save[step_index, :] = q_sol
    u_save[step_index, :] = u_sol * 1000.0
    pcc_position_save[step_index, :] = pcc_section_end_positions(q_sol)
    return q_sol


def save_results(
    y_expm: np.ndarray,
    t_list: np.ndarray,
    q_save: np.ndarray,
    u_save: np.ndarray,
    pcc_position_save: np.ndarray,
) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    np.save(PLOT_DIR / "y_expm", y_expm)
    np.save(PLOT_DIR / "t_expm", t_list)
    np.save(PLOT_DIR / "q_save", q_save)
    np.save(PLOT_DIR / "u_save", u_save)
    np.save(PLOT_DIR / "pcc_section_positions", pcc_position_save)


def main() -> None:
    sdk = connect_sdk()
    context = None
    socket = None

    try:
        desired_positions, t_list = load_trajectory()
        step_count = desired_positions.shape[1]

        y_expm = np.zeros((step_count, 6))
        q_save = np.zeros((step_count, 4))
        u_save = np.zeros((step_count, 6))
        pcc_position_save = np.zeros((step_count, 6))

        reference = initialize_motion_reference(sdk)
        context, socket = create_publisher(MOTOR_ENDPOINT)
        print("ZeroMQ publisher started on /controller/setActuatorDate")

        q0 = None
        for step_index in range(step_count):
            start_time = time.perf_counter()
            q0 = run_control_step(
                sdk=sdk,
                socket=socket,
                step_index=step_index,
                desired_position=desired_positions[:, step_index],
                q0=q0,
                reference=reference,
                y_expm=y_expm,
                q_save=q_save,
                u_save=u_save,
                pcc_position_save=pcc_position_save,
            )
            elapsed_time = time.perf_counter() - start_time
            print(f"Step {step_index}: running time = {elapsed_time:.6f}s\n")

        save_results(y_expm, t_list[:step_count], q_save, u_save, pcc_position_save)
        print(y_expm)

    finally:
        if socket is not None and context is not None:
            close_publisher(context, socket)
        sdk.Close()


if __name__ == "__main__":
    main()
