from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sciIO


BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent
TRAJ_DIR = BASE_DIR / "traj_data"


def load_experiment_results():
    zd = sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"].T * 1000.0
    y_expm = np.load(PLOT_DIR / "y_expm.npy")
    t_expm = np.squeeze(np.load(PLOT_DIR / "t_expm.npy"))
    u_save = np.load(PLOT_DIR / "u_save.npy")

    step_count = min(len(t_expm), y_expm.shape[0], zd.shape[0], u_save.shape[0])
    return zd[:step_count], y_expm[:step_count], t_expm[:step_count], u_save[:step_count]


def plot_tracking_results(t_expm, y_expm, zd):
    fig = plt.figure()
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["x", "y", "z"]

    for axis_idx, ax in enumerate(axes):
        ax.plot(t_expm, y_expm[:, axis_idx], label=f"measured {labels[axis_idx]}")
        ax.plot(t_expm, zd[:, axis_idx], label=f"desired {labels[axis_idx]}")
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
    zd, y_expm, t_expm, u_save = load_experiment_results()
    plot_tracking_results(t_expm, y_expm, zd)
    plot_motor_commands(t_expm, u_save)


if __name__ == "__main__":
    main()
