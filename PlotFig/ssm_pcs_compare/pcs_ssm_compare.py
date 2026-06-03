import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


PLOT_DIR = Path(__file__).resolve().parent
PCS_RESULT_DIRS = (PLOT_DIR / "pcs_results", PLOT_DIR / "pcs_results_ok")
SSM_RESULT_DIRS = (PLOT_DIR / "SSM_results", PLOT_DIR / "SSM_results_ok")
REFERENCE_TRAJ_FILES = {
    "eight": PLOT_DIR / "reference_zd_eight.npy",
    "spiral": PLOT_DIR / "reference_zd_spiral.npy",
}
# DEFAULT_TRAJECTORY = "spiral"
DEFAULT_TRAJECTORY = "eight"
DEFAULT_ERROR_METRIC = "norm"
# --error-metric norm   # 原来的瞬时误差范数，默认
# --error-metric ise    # 累积积分平方误差 ISE
# --error-metric both   # 两种误差图都画

def resolve_result_dir(candidate_dirs, trajectory: str, required_files) -> Path:
    traj_key = trajectory.lower()
    checked_dirs = []

    for base_dir in candidate_dirs:
        base_dir = Path(base_dir)
        search_dirs = (base_dir / traj_key, base_dir)
        for result_dir in search_dirs:
            checked_dirs.append(result_dir)
            if all((result_dir / name).exists() for name in required_files):
                return result_dir

    checked = "\n  ".join(str(path) for path in checked_dirs)
    raise FileNotFoundError(
        f"Could not find result files for trajectory '{traj_key}'. Checked:\n  {checked}"
    )


def load_reference_trajectory(trajectory: str):
    traj_key = trajectory.lower()
    if traj_key not in REFERENCE_TRAJ_FILES:
        choices = ", ".join(sorted(REFERENCE_TRAJ_FILES))
        raise ValueError(f"Unknown trajectory '{trajectory}'. Choose one of: {choices}")
    ref_path = REFERENCE_TRAJ_FILES[traj_key]
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing local reference trajectory file: {ref_path}")
    return np.load(ref_path)


def trim_to_common_length(*arrays):
    step_count = min(array.shape[0] for array in arrays)
    return tuple(array[:step_count] for array in arrays)


def load_pcs_results(trajectory: str):
    required_files = (
        "t_expm_pcs.npy",
        "y_expm_pcs.npy",
        "u_save_pcs.npy",
        "pcs_section_positions_pcs.npy",
    )
    result_dir = resolve_result_dir(PCS_RESULT_DIRS, trajectory, required_files)
    print(f"Loading PCS {trajectory.lower()} results from {result_dir}")

    t_pcs = np.squeeze(np.load(result_dir / "t_expm_pcs.npy"))
    y_pcs = np.load(result_dir / "y_expm_pcs.npy")
    u_pcs = np.load(result_dir / "u_save_pcs.npy")
    pcs_positions = np.load(result_dir / "pcs_section_positions_pcs.npy")
    return trim_to_common_length(t_pcs, y_pcs, u_pcs, pcs_positions)


def load_ssm_results(trajectory: str):
    required_files = ("t_ssm.npy", "y_exprm_ssm.npy", "ut_save_ssm.npy")
    result_dir = resolve_result_dir(SSM_RESULT_DIRS, trajectory, required_files)
    print(f"Loading SSM {trajectory.lower()} results from {result_dir}")

    t_ssm = np.squeeze(np.load(result_dir / "t_ssm.npy"))
    y_ssm = np.load(result_dir / "y_exprm_ssm.npy")
    u_ssm = np.load(result_dir / "ut_save_ssm.npy")
    return trim_to_common_length(t_ssm, y_ssm, u_ssm)


def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def invert_z_axis(ax):
    z_lower, z_upper = ax.get_zlim3d()
    ax.set_zlim3d(z_upper, z_lower)


def plot_tracking_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd):
    fig = plt.figure()
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["x", "y", "z"]
    pcs_count = min(len(t_pcs), len(y_pcs), len(zd))
    ssm_count = min(len(t_ssm), len(y_ssm), len(zd))

    for axis_idx, ax in enumerate(axes):
        ax.plot(t_pcs[:pcs_count], y_pcs[:pcs_count, axis_idx], label=f"PCS measured {labels[axis_idx]}")
        ax.plot(t_ssm[:ssm_count], y_ssm[:ssm_count, axis_idx], label=f"SSM measured {labels[axis_idx]}")
        ax.plot(t_pcs[:pcs_count], zd[:pcs_count, axis_idx], "k--", label=f"desired {labels[axis_idx]}")
        ax.set_ylabel("Position (mm)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("PCS vs SSM Tracking Results")
    plt.show()


def cumulative_trapezoid(values, time):
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    if values.size == 0:
        return values
    if values.size == 1:
        return np.zeros_like(values)
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    return np.concatenate(([0.0], np.cumsum(increments)))


def tracking_error_norm(y_measured, zd):
    return np.linalg.norm(y_measured[:, :3] - zd[:, :3], axis=1)


def plot_error_norm_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd):
    pcs_count = min(len(t_pcs), len(y_pcs), len(zd))
    ssm_count = min(len(t_ssm), len(y_ssm), len(zd))
    pcs_error = tracking_error_norm(y_pcs[:pcs_count], zd[:pcs_count])
    ssm_error = tracking_error_norm(y_ssm[:ssm_count], zd[:ssm_count])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_pcs[:pcs_count], pcs_error, label=f"PCS error, RMS={rms(pcs_error):.2f} mm")
    ax.plot(t_ssm[:ssm_count], ssm_error, label=f"SSM error, RMS={rms(ssm_error):.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking error norm (mm)")
    ax.set_title("PCS vs SSM Tracking Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_ise_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd):
    pcs_count = min(len(t_pcs), len(y_pcs), len(zd))
    ssm_count = min(len(t_ssm), len(y_ssm), len(zd))
    pcs_squared_error = tracking_error_norm(y_pcs[:pcs_count], zd[:pcs_count]) ** 2
    ssm_squared_error = tracking_error_norm(y_ssm[:ssm_count], zd[:ssm_count]) ** 2
    pcs_ise = cumulative_trapezoid(pcs_squared_error, t_pcs[:pcs_count])
    ssm_ise = cumulative_trapezoid(ssm_squared_error, t_ssm[:ssm_count])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_pcs[:pcs_count], pcs_ise, label=f"PCS ISE={pcs_ise[-1]:.2f} mm^2*s")
    ax.plot(t_ssm[:ssm_count], ssm_ise, label=f"SSM ISE={ssm_ise[-1]:.2f} mm^2*s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative ISE (mm^2*s)")
    ax.set_title("PCS vs SSM Integral Squared Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_error_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd, error_metric: str = DEFAULT_ERROR_METRIC):
    metric = error_metric.lower()
    if metric in ("norm", "both"):
        plot_error_norm_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd)
    if metric in ("ise", "both"):
        plot_ise_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd)


def plot_motor_commands_compare(t_pcs, u_pcs, t_ssm, u_ssm):
    fig = plt.figure()
    axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]
    labels = ["long 1", "long 2", "long 3", "short 1", "short 2", "short 3"]

    motor_count = min(6, u_pcs.shape[1], u_ssm.shape[1])
    for motor_idx in range(motor_count):
        ax = axes[motor_idx]
        ax.plot(t_pcs, u_pcs[:, motor_idx], label="PCS")
        ax.plot(t_ssm, u_ssm[:, motor_idx], label="SSM")
        ax.set_title(labels[motor_idx])
        ax.set_ylabel("Command (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[-2:]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("PCS vs SSM Tendon Commands")
    plt.show()


def plot_3d_tracking_compare(y_pcs, y_ssm, zd):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(y_pcs[:, 0], y_pcs[:, 1], y_pcs[:, 2], label="PCS measured end")
    ax.plot(y_ssm[:, 0], y_ssm[:, 1], y_ssm[:, 2], label="SSM measured end")
    ax.plot(zd[:, 0], zd[:, 1], zd[:, 2], "k--", label="desired end")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("PCS vs SSM 3D Tracking")
    set_axes_equal_3d(ax)
    invert_z_axis(ax)
    ax.legend()
    plt.show()


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def plot_pcs_ssm_compare(trajectory: str = DEFAULT_TRAJECTORY, error_metric: str = DEFAULT_ERROR_METRIC):
    trajectory = trajectory.lower()
    zd = load_reference_trajectory(trajectory)
    t_pcs, y_pcs, u_pcs, _pcs_positions = load_pcs_results(trajectory)
    t_ssm, y_ssm, u_ssm = load_ssm_results(trajectory)

    zd_for_3d = zd[: min(len(zd), len(y_pcs), len(y_ssm))]
    plot_tracking_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd)
    plot_error_compare(t_pcs, y_pcs, t_ssm, y_ssm, zd, error_metric)
    plot_motor_commands_compare(t_pcs, u_pcs, t_ssm, u_ssm)
    plot_3d_tracking_compare(y_pcs[: len(zd_for_3d)], y_ssm[: len(zd_for_3d)], zd_for_3d)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PCS and SSM tracking results.")
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=sorted(REFERENCE_TRAJ_FILES),
        default=DEFAULT_TRAJECTORY,
        help="Reference trajectory used by the PCS and SSM runs.",
    )
    parser.add_argument(
        "--error-metric",
        "-e",
        choices=("norm", "ise", "both"),
        default=DEFAULT_ERROR_METRIC,
        help="Error comparison plot: instantaneous norm, cumulative ISE, or both.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_pcs_ssm_compare(trajectory=args.trajectory, error_metric=args.error_metric)
