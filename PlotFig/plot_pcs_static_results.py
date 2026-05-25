import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PLOT_DIR / "pcs_static_results"
TRAJ_DIRS = {
    "eight": BASE_DIR / "dynamics" / "trajinfom" / "eight",
    "spiral": BASE_DIR / "dynamics" / "trajinfom" / "spiral",
}
DEFAULT_TRAJECTORY = "spiral"
PCS_STATIC_UNIT_COUNT = 4


def load_static_results(trajectory: str):
    traj_key = trajectory.lower()
    if traj_key not in TRAJ_DIRS:
        choices = ", ".join(sorted(TRAJ_DIRS))
        raise ValueError(f"Unknown trajectory '{trajectory}'. Choose one of: {choices}")

    required_files = (
        "y_expm_pcs_static.npy",
        "t_expm_pcs_static.npy",
        "u_save_pcs_static.npy",
        "q_save_pcs_static.npy",
        "lambda_save_pcs_static.npy",
        "static_tip_pcs_static.npy",
    )
    missing = [name for name in required_files if not (RESULT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing PCS static result files in {RESULT_DIR}: {', '.join(missing)}"
        )

    zd = sciIO.loadmat(str(TRAJ_DIRS[traj_key] / "zd_proj.mat"))["zd_proj"].T * 1000.0
    y_expm = np.load(RESULT_DIR / "y_expm_pcs_static.npy")
    t_expm = np.squeeze(np.load(RESULT_DIR / "t_expm_pcs_static.npy"))
    u_save = np.load(RESULT_DIR / "u_save_pcs_static.npy")
    q_save = np.load(RESULT_DIR / "q_save_pcs_static.npy")
    lambda_save = np.load(RESULT_DIR / "lambda_save_pcs_static.npy")
    static_tip = np.load(RESULT_DIR / "static_tip_pcs_static.npy") * 1000.0

    step_count = min(
        len(t_expm),
        y_expm.shape[0],
        zd.shape[0],
        u_save.shape[0],
        q_save.shape[0],
        lambda_save.shape[0],
        static_tip.shape[0],
    )
    return (
        zd[:step_count],
        y_expm[:step_count],
        t_expm[:step_count],
        u_save[:step_count],
        q_save[:step_count],
        lambda_save[:step_count],
        static_tip[:step_count],
    )


def plot_tracking_results(t_expm, y_expm, zd, static_tip, trajectory: str):
    fig = plt.figure(f"PCS static {trajectory} tracking")
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["x", "y", "z"]

    for axis_idx, ax in enumerate(axes):
        ax.plot(t_expm, y_expm[:, axis_idx], label=f"measured {labels[axis_idx]}")
        ax.plot(t_expm, zd[:, axis_idx], "k--", label=f"desired {labels[axis_idx]}")
        ax.plot(t_expm, static_tip[:, axis_idx], ":", label=f"static model {labels[axis_idx]}")
        ax.set_ylabel("Position (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"PCS static {trajectory} tracking results")
    fig.tight_layout()


def plot_tracking_error(t_expm, y_expm, zd, static_tip, trajectory: str):
    measured_error = np.linalg.norm(y_expm[:, :3] - zd[:, :3], axis=1)
    static_error = np.linalg.norm(static_tip[:, :3] - zd[:, :3], axis=1)

    fig = plt.figure(f"PCS static {trajectory} tracking error")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_expm, measured_error, label=f"measured RMS={rms(measured_error):.2f} mm")
    ax.plot(t_expm, static_error, label=f"static model RMS={rms(static_error):.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking error norm (mm)")
    ax.set_title(f"PCS static {trajectory} tracking error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def plot_motor_commands(t_expm, u_save, trajectory: str):
    fig = plt.figure(f"PCS static {trajectory} motor commands")
    ax = fig.add_subplot(1, 1, 1)
    styles = ["r-", "g-", "b-", "r--", "g--", "b--"]
    labels = ["section 1 cable 1", "section 1 cable 2", "section 1 cable 3",
              "section 2 cable 1", "section 2 cable 2", "section 2 cable 3"]

    motor_count = min(u_save.shape[1], len(labels))
    for motor_idx in range(motor_count):
        ax.plot(t_expm, u_save[:, motor_idx], styles[motor_idx], label=labels[motor_idx])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tendon command in solver order (mm)")
    ax.set_title(f"PCS static {trajectory} motor commands")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def plot_cable_multipliers(t_expm, lambda_save, trajectory: str):
    fig = plt.figure(f"PCS static {trajectory} cable multipliers")
    ax = fig.add_subplot(1, 1, 1)
    labels = ["section 1 cable 1", "section 1 cable 2", "section 1 cable 3",
              "section 2 cable 1", "section 2 cable 2", "section 2 cable 3"]

    cable_count = min(lambda_save.shape[1], len(labels))
    for cable_idx in range(cable_count):
        ax.plot(t_expm, lambda_save[:, cable_idx], label=labels[cable_idx])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Length constraint multiplier")
    ax.set_title(f"PCS static {trajectory} cable multipliers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def plot_strain_states(t_expm, q_save, trajectory: str):
    fig = plt.figure(f"PCS static {trajectory} strain states")
    if q_save.shape[1] % PCS_STATIC_UNIT_COUNT != 0:
        raise ValueError(
            f"q_save column count must be divisible by {PCS_STATIC_UNIT_COUNT}, got {q_save.shape[1]}"
        )
    strain_size = q_save.shape[1] // PCS_STATIC_UNIT_COUNT
    if strain_size != 6:
        raise ValueError(f"Unsupported PCS static strain size: {strain_size}")

    state_labels = ["kx", "ky", "kz", "vx", "vy", "vz"]
    styles = ["r-", "g-", "b-", "k-", "c-", "m-"]
    axes = [fig.add_subplot(PCS_STATIC_UNIT_COUNT, 1, i + 1) for i in range(PCS_STATIC_UNIT_COUNT)]

    for unit_idx, ax in enumerate(axes):
        section_q = q_save[:, unit_idx * strain_size : (unit_idx + 1) * strain_size]
        for state_idx, (label, style) in enumerate(zip(state_labels, styles)):
            ax.plot(t_expm, section_q[:, state_idx], style, label=f"{label}{unit_idx + 1}")
        ax.set_ylabel(f"Unit {unit_idx + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"PCS static {trajectory} strain states")
    fig.tight_layout()


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def plot_pcs_static_results(trajectory: str, show_states: bool = False):
    zd, y_expm, t_expm, u_save, q_save, lambda_save, static_tip = load_static_results(trajectory)
    plot_tracking_results(t_expm, y_expm, zd, static_tip, trajectory)
    plot_tracking_error(t_expm, y_expm, zd, static_tip, trajectory)
    plot_motor_commands(t_expm, u_save, trajectory)
    plot_cable_multipliers(t_expm, lambda_save, trajectory)
    if show_states:
        plot_strain_states(t_expm, q_save, trajectory)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PCS static MPC tracking results.")
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=sorted(TRAJ_DIRS),
        default=DEFAULT_TRAJECTORY,
        help="Reference trajectory to compare against.",
    )
    parser.add_argument(
        "--show-states",
        action="store_true",
        help="Also plot static PCS strain states.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_pcs_static_results(args.trajectory, show_states=args.show_states)
    plt.show()


if __name__ == "__main__":
    main()
