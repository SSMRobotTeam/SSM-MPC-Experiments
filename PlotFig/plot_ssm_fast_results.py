import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamics.SSM.matlab_data import ROM_DIR, clone_struct_array, get_field, load_rom_data, subset_mapping_rows
from dynamics.SSM.ssm_dynamics import reduced_to_full


PLOT_DIR = Path(__file__).resolve().parent
FAST_RESULT_DIR = PLOT_DIR / "fast_results"
BASE_DIR = PLOT_DIR.parent
TRAJ_DIRS = {
    "eight": BASE_DIR / "dynamics" / "trajinfom" / "eight",
    "spiral": BASE_DIR / "dynamics" / "trajinfom" / "spiral",
}
DEFAULT_TRAJECTORY = "spiral"
SSM_FACTOR = 1.475
END_INITIAL_POSITION_M = np.array([0.0, 0.0, 0.4400])


def scaled_w0(rom_data):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR
    return w0


def w0_theoretical_output(pk_save, trajectory_dir):
    rom_data = load_rom_data(ROM_DIR, traj_dir=trajectory_dir)
    outdof_endeff = np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)
    outdof_half = np.asarray(rom_data["outdof_half"], dtype=int).reshape(-1)
    w0_for_out = subset_mapping_rows(scaled_w0(rom_data), np.concatenate([outdof_endeff, outdof_half]))
    mapped = reduced_to_full(np.asarray(pk_save, dtype=float).T, w0_for_out, None, 1)
    return mapped.T * 1000.0


def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def load_ssm_fast_results(plot_dir: str | Path = PLOT_DIR, trajectory: str = DEFAULT_TRAJECTORY):
    traj_key = trajectory.lower()
    if traj_key not in TRAJ_DIRS:
        choices = ", ".join(sorted(TRAJ_DIRS))
        raise ValueError(f"Unknown trajectory '{trajectory}'. Choose one of: {choices}")
    plot_dir = Path(plot_dir)
    if plot_dir == PLOT_DIR:
        plot_dir = FAST_RESULT_DIR / traj_key

    results = {
        "t": np.load(plot_dir / "t_ssm_fast.npy"),
        "y_exprm": np.load(plot_dir / "y_exprm_ssm_fast.npy"),
        "y_ssm": None,
        "y_w0": None,
        "ut_save": np.load(plot_dir / "ut_save_ssm_fast.npy"),
        "zd": sciIO.loadmat(str(TRAJ_DIRS[traj_key] / "zd_proj.mat"))["zd_proj"],
    }
    y_ssm_path = plot_dir / "y_ssm_ssm_fast.npy"
    if y_ssm_path.exists():
        results["y_ssm"] = np.load(y_ssm_path)
    else:
        mat_path = plot_dir / "ssm_python_results_fast.mat"
        if mat_path.exists():
            mat_data = sciIO.loadmat(str(mat_path), squeeze_me=True)
            if "y_ssm" in mat_data:
                results["y_ssm"] = np.asarray(mat_data["y_ssm"], dtype=float)
    pk_path = plot_dir / "pk_save_ssm_fast.npy"
    if pk_path.exists():
        results["y_w0"] = w0_theoretical_output(np.load(pk_path), TRAJ_DIRS[traj_key])
    return results


def plot_ssm_fast_results(results=None, trajectory: str = DEFAULT_TRAJECTORY):
    if results is None:
        results = load_ssm_fast_results(trajectory=trajectory)
    t = results["t"]
    y = results["y_exprm"]
    u = results["ut_save"]
    y_ssm = results.get("y_ssm")
    zd = results.get("zd")
    n = min(t.size, y.shape[0])

    fig1 = plt.figure()
    labels = ["end x", "end y", "end z"]
    for idx, label in enumerate(labels):
        ax = fig1.add_subplot(3, 1, idx + 1)
        ax.plot(t[:n], y[:n, idx], label=label)
        if zd is not None:
            ref_n = min(n, zd.shape[1])
            ax.plot(t[:ref_n], zd[idx, :ref_n] * 1000.0, "k--", label=f"desired {label}")
        ax.set_ylabel("Position (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig1.axes[-1].set_xlabel("Time (s)")
    fig1.tight_layout()

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    for idx in range(u.shape[1]):
        line_style = "-" if idx < 3 else "--"
        ax.plot(t[: u.shape[0]], u[:, idx], linestyle=line_style, label=f"u{idx + 1}")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor command (mm)")

    fig3 = plt.figure()
    ax = fig3.add_subplot(1, 1, 1, projection="3d")
    end_initial_mm = END_INITIAL_POSITION_M * 1000.0
    end_xyz = y[:n, :3] + end_initial_mm.reshape(1, 3)
    ax.plot(end_xyz[:, 0], end_xyz[:, 1], end_xyz[:, 2], label="measured end")
    ax.scatter(
        end_initial_mm[0],
        end_initial_mm[1],
        end_initial_mm[2],
        c="tab:green",
        marker="^",
        s=45,
        label="initial end",
    )
    # if n > 1:
    #     arrow_start_idx = int(np.argmin(np.abs(t[:n] - 0.0)))
    #     arrow_end_idx = int(np.argmin(np.abs(t[:n] - 2.0)))
    #     arrow_vector = end_xyz[arrow_end_idx] - end_xyz[arrow_start_idx]
    #     arrow_vector[2] = 0.0
    #     arrow_length_mm = np.linalg.norm(arrow_vector)
    # else:
    #     arrow_vector = np.zeros(3)
    #     arrow_length_mm = 0.0
    # if arrow_length_mm > 1e-9:
    #     arrow_direction = arrow_vector / arrow_length_mm
    #     normal_a = np.array([-arrow_direction[1], arrow_direction[0], 0.0])
    #     normal_b = -normal_a
    #     trajectory_center_xy = np.mean(end_xyz[:, :2], axis=0)
    #     arrow_anchor = end_initial_mm.copy()
    #     offset_length_mm = 10.0
    #     candidate_a = arrow_anchor + offset_length_mm * normal_a
    #     candidate_b = arrow_anchor + offset_length_mm * normal_b
    #     dist_a = np.linalg.norm(candidate_a[:2] - trajectory_center_xy)
    #     dist_b = np.linalg.norm(candidate_b[:2] - trajectory_center_xy)
    #     arrow_anchor = candidate_a if dist_a >= dist_b else candidate_b
    #     ax.quiver(
    #         arrow_anchor[0],
    #         arrow_anchor[1],
    #         arrow_anchor[2],
    #         arrow_vector[0],
    #         arrow_vector[1],
    #         arrow_vector[2],
    #         length=1.0,
    #         normalize=False,
    #         color="tab:red",
    #         linewidth=2.0,
    #         arrow_length_ratio=0.5,
    #         label="direction",
    #     )
    # if y_ssm is not None:
    #     ssm_n = min(n, y_ssm.shape[0])
    #     ssm_end_xyz = y_ssm[:ssm_n, :3] + end_initial_mm.reshape(1, 3)
    #     ax.plot(
    #         ssm_end_xyz[:, 0],
    #         ssm_end_xyz[:, 1],
    #         ssm_end_xyz[:, 2],
    #         color="tab:orange",
    #         label="SSM calculated end",
    #     )
    if zd is not None:
        ref_n = min(n, zd.shape[1])
        desired_xyz = zd[:3, :ref_n].T * 1000.0 + end_initial_mm.reshape(1, 3)
        ax.plot(
            desired_xyz[:, 0],
            desired_xyz[:, 1],
            desired_xyz[:, 2],
            "k--",
            label="desired end",
        )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    set_axes_equal_3d(ax)
    ax.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SSM fast tracking and motor command results.")
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=sorted(TRAJ_DIRS),
        default=DEFAULT_TRAJECTORY,
        help="Reference trajectory used by the fast run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ssm_fast_results(trajectory=args.trajectory)
