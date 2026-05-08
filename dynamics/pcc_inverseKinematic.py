import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO
from scipy.optimize import least_squares


SMALL_ANGLE = 1e-8
BASE_DIR = Path(__file__).resolve().parent.parent
TRAJ_DIR = BASE_DIR / "traj_data"


def _section_tip(theta: float, phi: float, length: float) -> np.ndarray:
    """PCC section tip position in its local frame."""
    if abs(theta) < SMALL_ANGLE:
        return np.array([0.0, 0.0, length])

    return (length / theta) * np.array(
        [
            np.cos(phi) * (1.0 - np.cos(theta)),
            np.sin(phi) * (1.0 - np.cos(theta)),
            np.sin(theta),
        ]
    )


def _section_rotation(theta: float, phi: float) -> np.ndarray:
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_phi, s_phi = np.cos(phi), np.sin(phi)

    rz = np.array(
        [
            [c_phi, -s_phi, 0.0],
            [s_phi, c_phi, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [c_theta, 0.0, s_theta],
            [0.0, 1.0, 0.0],
            [-s_theta, 0.0, c_theta],
        ]
    )
    return rz @ ry


def config_to_work(q, L1, L2):
    """Forward kinematics: configuration q -> end-effector position."""
    theta1, phi1, theta2, phi2 = np.asarray(q, dtype=float)

    p1 = _section_tip(theta1, phi1, L1)
    p2_local = _section_tip(theta2, phi2, L2)
    return p1 + _section_rotation(theta1, phi1) @ p2_local


def config_to_tendon(q, d1, d2):
    """Map configuration q = [theta1, phi1, theta2, phi2] to tendon lengths."""
    theta1, phi1, theta2, phi2 = np.asarray(q, dtype=float)

    section_2_phi = phi1 + phi2
    long_angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    short_angles = np.array([np.pi, np.pi / 3.0, 5.0 * np.pi / 3.0])

    long_tendons = (
        - d1 * theta1 * np.cos(long_angles - phi1)
        - d2 * theta2 * np.cos(long_angles - section_2_phi)
    )
    short_tendons = -d1 * theta1 * np.cos(short_angles - phi1)
    return np.array([*long_tendons, *short_tendons])


def ik_position(x_d, L1, L2, d1, d2, q0=None, R_config=1e-2, R_work=1e8):
    """Inverse kinematics: desired position -> configuration and tendon lengths."""
    x_d = np.asarray(x_d, dtype=float).reshape(3)
    q_init = np.array([SMALL_ANGLE] * 4, dtype=float) if q0 is None else np.asarray(q0, dtype=float)

    def residuals(q):
        xe = config_to_work(q, L1, L2)
        r_pos = xe - x_d
        r_reg = np.array([q[0], q[2]])
        return np.hstack([R_work * r_pos, R_config * r_reg])

    res = least_squares(
        residuals,
        q_init,
        method="lm",
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=1000,
    )
    q_sol = res.x
    u_sol = config_to_tendon(q_sol, d1, d2)
    return u_sol, q_sol, res


def plot_pcc_control_results(t_list, u_save, q_save, xe_save, xd_list):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1, projection="3d")
    ax1.plot(xd_list[0, :], xd_list[1, :], xd_list[2, :], "r--", label="desired trajectory")
    ax1.plot(xe_save[0, :], xe_save[1, :], xe_save[2, :], "b-", label="achieved trajectory")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1])

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(2, 1, 1)
    labels = ["u1", "u2", "u3", "u4", "u5", "u6"]
    styles = ["r-", "g-", "b-", "r--", "g--", "b--"]
    for row, label, style in zip(u_save, labels, styles):
        ax2.plot(t_list, row * 1000.0, style, label=label)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tendon lengths (mm)")
    ax2.legend()

    ax3 = fig2.add_subplot(2, 1, 2)
    for row, label in zip(q_save, ["theta1", "phi1", "theta2", "phi2"]):
        ax3.plot(t_list, row, label=label)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Configuration (rad)")
    ax3.legend()
    plt.show()


def run_offline_demo():
    L1, L2 = 0.225, 0.215
    d1, d2 = 0.015, 0.015
    x0 = np.array([[0.0], [0.0], [L1 + L2]])

    xd_list = sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"] + x0
    t_list = sciIO.loadmat(str(TRAJ_DIR / "t_list.mat"))["t_list"].squeeze()

    _, num_steps = xd_list.shape
    q0 = None
    u_save = np.zeros((6, num_steps))
    q_save = np.zeros((4, num_steps))
    xe_save = np.zeros((3, num_steps))

    for i in range(num_steps):
        start_time = time.perf_counter()
        u_sol, q_sol, _ = ik_position(xd_list[:, i], L1, L2, d1, d2, q0=q0)
        q0 = q_sol
        u_save[:, i] = u_sol
        q_save[:, i] = q_sol
        xe_save[:, i] = config_to_work(q_sol, L1, L2)
        print(f"step {i} running time: {time.perf_counter() - start_time:.6f} s")

    plot_pcc_control_results(t_list, u_save, q_save, xe_save, xd_list)


if __name__ == "__main__":
    run_offline_demo()
