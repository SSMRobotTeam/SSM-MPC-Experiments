from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
# TRAJ_DIR = BASE_DIR / "traj_data"
TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/spiral"
# TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/eight"

L1, L2 = 0.2225, 0.2175
SECTION_1_OFFSET = np.array([0.0, 0.0, L1])
SECTION_2_OFFSET = np.array([0.0, 0.0, L1 + L2])
PCS_UNIT_COUNT = 4


def load_experiment_results():
    zd = sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"].T * 1000.0
    y_expm = np.load(PLOT_DIR / "y_expm_pcs.npy")
    t_expm = np.squeeze(np.load(PLOT_DIR / "t_expm_pcs.npy"))
    u_save = np.load(PLOT_DIR / "u_save_pcs.npy")
    q_save = np.load(PLOT_DIR / "q_save_pcs.npy")
    pcs_positions = np.load(PLOT_DIR / "pcs_section_positions_pcs.npy")

    step_count = min(
        len(t_expm),
        y_expm.shape[0],
        zd.shape[0],
        u_save.shape[0],
        q_save.shape[0],
        pcs_positions.shape[0],
    )
    return (
        zd[:step_count],
        y_expm[:step_count],
        t_expm[:step_count],
        u_save[:step_count],
        q_save[:step_count],
        pcs_positions[:step_count],
    )


def pcs_positions_to_displacements(pcs_positions):
    section_1 = (pcs_positions[:, :3] - SECTION_1_OFFSET.reshape(1, 3)) * 1000.0
    section_2 = (pcs_positions[:, 3:6] - SECTION_2_OFFSET.reshape(1, 3)) * 1000.0
    return section_1, section_2


def plot_tracking_results(t_expm, y_expm, zd, pcs_positions):
    fig = plt.figure()
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["x", "y", "z"]
    pcs_section_1, pcs_section_2 = pcs_positions_to_displacements(pcs_positions)

    for axis_idx, ax in enumerate(axes):
        ax.plot(t_expm, y_expm[:, axis_idx], label=f"measured {labels[axis_idx]}")
        ax.plot(t_expm, zd[:, axis_idx], label=f"desired {labels[axis_idx]}")
        # ax.plot(t_expm, pcs_section_1[:, axis_idx], "--", label=f"pcs section 1 {labels[axis_idx]}")
        ax.plot(t_expm, pcs_section_2[:, axis_idx], ":", label=f"pcs section 2 {labels[axis_idx]}")
        ax.set_ylabel("Position (mm)")
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.show()


def plot_motor_commands(t_expm, u_save):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    styles = ["r-", "g-", "b-", "r--", "g--", "b--"]
    labels = ["long 1", "long 2", "long 3", "short 1", "short 2", "short 3"]

    for motor_idx, (style, label) in enumerate(zip(styles, labels)):
        ax.plot(t_expm, u_save[:, motor_idx], style, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tendon command in solver order (mm)")
    ax.legend()
    plt.show()


def plot_strain_states(t_expm, q_save):
    fig = plt.figure()
    if q_save.shape[1] % PCS_UNIT_COUNT != 0:
        raise ValueError(f"q_save column count must be divisible by {PCS_UNIT_COUNT}, got {q_save.shape[1]}")
    strain_size = q_save.shape[1] // PCS_UNIT_COUNT
    if strain_size == 4:
        state_labels = ["kx", "ky", "tau", "eps"]
        styles = ["r-", "g-", "b-", "k-"]
    elif strain_size == 6:
        state_labels = ["kx", "ky", "tau", "eps", "gx", "gy"]
        styles = ["r-", "g-", "b-", "k-", "c-", "m-"]
    else:
        raise ValueError(f"Unsupported PCS strain size: {strain_size}")
    unit_count = PCS_UNIT_COUNT
    axes = [fig.add_subplot(unit_count, 1, i + 1) for i in range(unit_count)]

    for unit_idx, ax in enumerate(axes):
        section_q = q_save[:, unit_idx * strain_size : (unit_idx + 1) * strain_size]
        for state_idx, (label, style) in enumerate(zip(state_labels, styles)):
            ax.plot(t_expm, section_q[:, state_idx], style, label=f"{label}{unit_idx + 1}")
        ax.set_ylabel(f"Unit {unit_idx + 1}")
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.show()


def main():
    zd, y_expm, t_expm, u_save, q_save, pcs_positions = load_experiment_results()
    plot_tracking_results(t_expm, y_expm, zd, pcs_positions)
    plot_motor_commands(t_expm, u_save)
    # plot_strain_states(t_expm, q_save)


if __name__ == "__main__":
    main()
