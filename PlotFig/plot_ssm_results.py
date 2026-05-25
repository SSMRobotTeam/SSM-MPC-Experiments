import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLOT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PLOT_DIR / "SSM_results"
BASE_DIR = PLOT_DIR.parent
TRAJ_DIRS = {
    "eight": BASE_DIR / "dynamics" / "trajinfom" / "eight",
    "spiral": BASE_DIR / "dynamics" / "trajinfom" / "spiral",
}
DEFAULT_TRAJECTORY = "spiral"


def resolve_result_dir(plot_dir: str | Path, trajectory: str) -> Path:
    traj_key = trajectory.lower()
    base_dir = Path(plot_dir)
    if base_dir.name.lower() == traj_key:
        return base_dir
    if base_dir.name == RESULT_DIR.name or base_dir.resolve() == RESULT_DIR.resolve():
        return base_dir / traj_key
    return base_dir


def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def load_experiment_results(plot_dir: str | Path = RESULT_DIR, trajectory: str = DEFAULT_TRAJECTORY):
    traj_key = trajectory.lower()
    if traj_key not in TRAJ_DIRS:
        choices = ", ".join(sorted(TRAJ_DIRS))
        raise ValueError(f"Unknown trajectory '{trajectory}'. Choose one of: {choices}")
    plot_dir = resolve_result_dir(plot_dir, traj_key)
    required_files = ("t_ssm.npy", "y_exprm_ssm.npy", "ut_save_ssm.npy")
    missing_files = [name for name in required_files if not (plot_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing SSM result files in {plot_dir}: {', '.join(missing_files)}. "
            f"Expected files under {RESULT_DIR / traj_key} for trajectory '{traj_key}'."
        )
    print(f"Loading SSM {traj_key} results from {plot_dir}")
    zd = sciIO.loadmat(str(TRAJ_DIRS[traj_key] / "zd_proj.mat"))["zd_proj"].T * 1000.0
    y_expm = np.load(plot_dir / "y_exprm_ssm.npy")
    t_expm = np.squeeze(np.load(plot_dir / "t_ssm.npy"))
    u_save = np.load(plot_dir / "ut_save_ssm.npy")

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
    labels = ["long 1", "long 2", "long 3", "short 1", "short 2", "short 3"]

    for motor_idx, (style, label) in enumerate(zip(styles, labels)):
        ax.plot(t_expm, u_save[:, motor_idx], style, label=label)

    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tendon command in solver order (mm)")
    plt.show()


def plot_3d_tracking_results(y_expm, zd):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    end_xyz = y_expm[:, :3]
    ax.plot(end_xyz[:, 0], end_xyz[:, 1], end_xyz[:, 2], label="measured end")
    ax.plot(zd[:, 0], zd[:, 1], zd[:, 2], "k--", label="desired end")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    set_axes_equal_3d(ax)
    ax.legend()
    plt.show()


def plot_ssm_results(trajectory: str = DEFAULT_TRAJECTORY):
    zd, y_expm, t_expm, u_save = load_experiment_results(trajectory=trajectory)
    plot_tracking_results(t_expm, y_expm, zd)
    plot_motor_commands(t_expm, u_save)
    plot_3d_tracking_results(y_expm, zd)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SSM tracking and motor command results.")
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=sorted(TRAJ_DIRS),
        default=DEFAULT_TRAJECTORY,
        help="Reference trajectory used by the SSM run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ssm_results(trajectory=args.trajectory)
