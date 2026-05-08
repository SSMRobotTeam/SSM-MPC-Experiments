from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sciIO


BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
# TRAJ_DIR = BASE_DIR / "traj_data"
# TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/spiral"
TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/eight"

L1, L2 = 0.2225, 0.2175
SECTION_1_OFFSET = np.array([0.0, 0.0, L1])
SECTION_2_OFFSET = np.array([0.0, 0.0, L1 + L2])


def load_experiment_results():
    zd = sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"].T * 1000.0
    y_expm = np.load(PLOT_DIR / "y_expm.npy")
    t_expm = np.squeeze(np.load(PLOT_DIR / "t_expm.npy"))
    u_save = np.load(PLOT_DIR / "u_save.npy")
    pcc_positions = np.load(PLOT_DIR / "pcc_section_positions.npy")

    step_count = min(len(t_expm), y_expm.shape[0], zd.shape[0], u_save.shape[0], pcc_positions.shape[0])
    return zd[:step_count], y_expm[:step_count], t_expm[:step_count], u_save[:step_count], pcc_positions[:step_count]


def pcc_positions_to_displacements(pcc_positions):
    section_1 = (pcc_positions[:, :3] - SECTION_1_OFFSET.reshape(1, 3)) * 1000.0
    section_2 = (pcc_positions[:, 3:6] - SECTION_2_OFFSET.reshape(1, 3)) * 1000.0
    return section_1, section_2


def plot_tracking_results(t_expm, y_expm, zd, pcc_positions):
    fig = plt.figure()
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["x", "y", "z"]
    pcc_section_1, pcc_section_2 = pcc_positions_to_displacements(pcc_positions)

    for axis_idx, ax in enumerate(axes):
        ax.plot(t_expm, y_expm[:, axis_idx], label=f"measured {labels[axis_idx]}")
        ax.plot(t_expm, zd[:, axis_idx], label=f"desired {labels[axis_idx]}")
        # ax.plot(t_expm, pcc_section_1[:, axis_idx], "--", label=f"pcc section 1 {labels[axis_idx]}")
        ax.plot(t_expm, pcc_section_2[:, axis_idx], ":", label=f"pcc section 2 {labels[axis_idx]}")
        ax.set_ylabel("Position (mm)")
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.show()


def plot_motor_commands(t_expm, u_save):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    styles = ["r-", "g-", "b-", "r--", "g--", "b--"]

    for motor_idx, style in enumerate(styles):
        ax.plot(t_expm, u_save[:, motor_idx], style, label=f"u{motor_idx + 1}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tendon command (mm)")
    ax.legend()
    plt.show()


def main():
    zd, y_expm, t_expm, u_save, pcc_positions = load_experiment_results()
    plot_tracking_results(t_expm, y_expm, zd, pcc_positions)
    plot_motor_commands(t_expm, u_save)


if __name__ == "__main__":
    main()
