import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


PLOT_DIR = Path(__file__).resolve().parent
BASE_DIR = PLOT_DIR.parent
SSM_RESULT_DIRS = (PLOT_DIR / "SSM_results", PLOT_DIR / "SSM_results_ok")
DATASSM_DIR = PLOT_DIR / "DataSSM"
TRAJ_DIRS = {
    "eight": BASE_DIR / "dynamics" / "trajinfom" / "eight",
    "spiral": BASE_DIR / "dynamics" / "trajinfom" / "spiral",
}
DEFAULT_TRAJECTORY = "eight"


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
        f"Could not find SSM result files for trajectory '{traj_key}'. Checked:\n  {checked}"
    )


def load_reference_trajectory(trajectory: str):
    traj_key = trajectory.lower()
    if traj_key not in TRAJ_DIRS:
        choices = ", ".join(sorted(TRAJ_DIRS))
        raise ValueError(f"Unknown trajectory '{trajectory}'. Choose one of: {choices}")
    return sciIO.loadmat(str(TRAJ_DIRS[traj_key] / "zd_proj.mat"))["zd_proj"].T * 1000.0


def trim_to_common_length(*arrays):
    step_count = min(array.shape[0] for array in arrays)
    return tuple(array[:step_count] for array in arrays)


def load_ssm_results(trajectory: str):
    required_files = ("t_ssm.npy", "y_exprm_ssm.npy", "ut_save_ssm.npy")
    result_dir = resolve_result_dir(SSM_RESULT_DIRS, trajectory, required_files)
    print(f"Loading SSM {trajectory.lower()} results from {result_dir}")

    t_ssm = np.squeeze(np.load(result_dir / "t_ssm.npy"))
    y_ssm = np.load(result_dir / "y_exprm_ssm.npy")
    u_ssm = np.load(result_dir / "ut_save_ssm.npy")
    return trim_to_common_length(t_ssm, y_ssm, u_ssm)


def load_datassm_results(trajectory: str, data_file: str | Path | None = None):
    traj_key = trajectory.lower()
    if data_file is None:
        matches = sorted(DATASSM_DIR.glob(f"*{traj_key}*.mat"))
        if not matches:
            raise FileNotFoundError(f"Could not find DataSSM .mat file for trajectory '{traj_key}' in {DATASSM_DIR}")
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise ValueError(f"Found multiple DataSSM files for '{traj_key}': {names}. Pass --data-file.")
        data_path = matches[0]
    else:
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = DATASSM_DIR / data_path

    print(f"Loading DataSSM {traj_key} results from {data_path}")
    data = sciIO.loadmat(str(data_path), squeeze_me=True)
    required_keys = ("t_actual", "phase", "y_end_model_frame", "u_sent_motor_posiSp", "ref")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"Missing DataSSM fields in {data_path}: {', '.join(missing)}")

    phase = np.asarray(data["phase"]).reshape(-1)
    open_loop_mask = np.array([str(value) == "open_loop" for value in phase], dtype=bool)
    if not np.any(open_loop_mask):
        open_loop_mask = np.ones_like(phase, dtype=bool)

    t_data = np.asarray(data["t_actual"], dtype=float).reshape(-1)[open_loop_mask]
    t_data = t_data - t_data[0]
    y_data = np.asarray(data["y_end_model_frame"], dtype=float)[open_loop_mask, :3]
    u_data = np.asarray(data["u_sent_motor_posiSp"], dtype=float)[open_loop_mask, :]
    ref = np.asarray(data["ref"], dtype=float).T

    return trim_to_common_length(t_data, y_data, u_data, ref)


def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def plot_tracking_compare(t_ssm, y_ssm, t_data, y_data, zd, trajectory: str):
    fig = plt.figure(f"SSM vs DataSSM {trajectory} tracking")
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["x", "y", "z"]
    ssm_count = min(len(t_ssm), len(y_ssm), len(zd))
    data_count = min(len(t_data), len(y_data), len(zd))

    for axis_idx, ax in enumerate(axes):
        ax.plot(t_ssm[:ssm_count], y_ssm[:ssm_count, axis_idx], label=f"SSM measured {labels[axis_idx]}")
        ax.plot(t_data[:data_count], y_data[:data_count, axis_idx], label=f"DataSSM measured {labels[axis_idx]}")
        ax.plot(t_ssm[:ssm_count], zd[:ssm_count, axis_idx], "k--", label=f"desired {labels[axis_idx]}")
        ax.set_ylabel("Position (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"SSM Results vs DataSSM Tracking ({trajectory})")
    fig.tight_layout()
    return fig


def plot_error_compare(t_ssm, y_ssm, t_data, y_data, zd, trajectory: str):
    ssm_count = min(len(t_ssm), len(y_ssm), len(zd))
    data_count = min(len(t_data), len(y_data), len(zd))
    ssm_error = np.linalg.norm(y_ssm[:ssm_count, :3] - zd[:ssm_count, :3], axis=1)
    data_error = np.linalg.norm(y_data[:data_count, :3] - zd[:data_count, :3], axis=1)

    fig = plt.figure(f"SSM vs DataSSM {trajectory} error")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_ssm[:ssm_count], ssm_error, label=f"SSM error, RMS={rms(ssm_error):.2f} mm")
    ax.plot(t_data[:data_count], data_error, label=f"DataSSM error, RMS={rms(data_error):.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking error norm (mm)")
    ax.set_title(f"SSM Results vs DataSSM Tracking Error ({trajectory})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_motor_commands_compare(t_ssm, u_ssm, t_data, u_data, trajectory: str):
    fig = plt.figure(f"SSM vs DataSSM {trajectory} tendon commands")
    axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]
    labels = ["long 1", "long 2", "long 3", "short 1", "short 2", "short 3"]

    motor_count = min(6, u_ssm.shape[1], u_data.shape[1])
    for motor_idx in range(motor_count):
        ax = axes[motor_idx]
        ax.plot(t_ssm[: u_ssm.shape[0]], u_ssm[:, motor_idx], label="SSM_results")
        ax.plot(t_data[: u_data.shape[0]], u_data[:, motor_idx], label="DataSSM")
        ax.set_title(labels[motor_idx])
        ax.set_ylabel("Command (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[-2:]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"SSM Results vs DataSSM Tendon Commands ({trajectory})")
    fig.tight_layout()
    return fig


def plot_3d_tracking_compare(y_ssm, y_data, zd, trajectory: str):
    fig = plt.figure(f"SSM vs DataSSM {trajectory} 3D tracking")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(y_ssm[:, 0], y_ssm[:, 1], y_ssm[:, 2], label="SSM_results measured end")
    ax.plot(y_data[:, 0], y_data[:, 1], y_data[:, 2], label="DataSSM measured end")
    ax.plot(zd[:, 0], zd[:, 1], zd[:, 2], "k--", label="desired end")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"SSM Results vs DataSSM 3D Tracking ({trajectory})")
    set_axes_equal_3d(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figures(figures, save_dir: str | Path, trajectory: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        output_path = save_dir / f"{trajectory}_{name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {output_path}")


def plot_ssm_datassm_compare(
    trajectory: str = DEFAULT_TRAJECTORY,
    data_file: str | Path | None = None,
    save_dir: str | Path | None = None,
    show: bool = True,
):
    if not show:
        plt.switch_backend("Agg")

    trajectory = trajectory.lower()
    reference_zd = load_reference_trajectory(trajectory)
    t_ssm, y_ssm, u_ssm = load_ssm_results(trajectory)
    t_data, y_data, u_data, data_ref = load_datassm_results(trajectory, data_file=data_file)

    # Prefer the DataSSM reference when present, because it is bundled with the measured run.
    zd = data_ref if data_ref.size else reference_zd
    ssm_for_3d_count = min(len(zd), len(y_ssm), len(y_data))
    figures = [
        ("tracking", plot_tracking_compare(t_ssm, y_ssm, t_data, y_data, zd, trajectory)),
        ("error", plot_error_compare(t_ssm, y_ssm, t_data, y_data, zd, trajectory)),
        ("motor_commands", plot_motor_commands_compare(t_ssm, u_ssm, t_data, u_data, trajectory)),
        (
            "3d_tracking",
            plot_3d_tracking_compare(
                y_ssm[:ssm_for_3d_count, :3],
                y_data[:ssm_for_3d_count, :3],
                zd[:ssm_for_3d_count, :3],
                trajectory,
            ),
        ),
    ]

    if save_dir is not None:
        save_figures(figures, save_dir, trajectory)
    if show:
        plt.show()
    return figures


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SSM_results and DataSSM tracking results.")
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=sorted(TRAJ_DIRS),
        default=DEFAULT_TRAJECTORY,
        help="Reference trajectory used by both SSM_results and DataSSM.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional DataSSM .mat file name or path. If omitted, match by trajectory name.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory for saving the comparison figures as PNG files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save or create figures without opening matplotlib windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ssm_datassm_compare(
        trajectory=args.trajectory,
        data_file=args.data_file,
        save_dir=args.save_dir,
        show=not args.no_show,
    )
