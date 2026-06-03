from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = SCRIPT_DIR / "DataVerify.mat"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
TIME_MIN = 0.0
TIME_MAX = 2.0
FONT_SIZE = 16


def _load_data_verify(mat_path: Path):
    mat_data = sciIO.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "DataVerify" not in mat_data:
        available = ", ".join(key for key in mat_data if not key.startswith("__"))
        raise KeyError(f"Missing variable 'DataVerify' in {mat_path}. Available: {available}")
    return mat_data["DataVerify"]


def _as_time_vector(values, name: str) -> np.ndarray:
    time = np.asarray(values, dtype=float).reshape(-1)
    if time.size == 0:
        raise ValueError(f"{name} is empty.")
    return time


def _as_xyz(values, sample_count: int, name: str) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        if data.size == 3 * sample_count:
            return data.reshape(sample_count, 3)
        raise ValueError(f"Cannot interpret {name} shape {data.shape} as xyz samples.")

    rows, cols = data.shape
    if rows == sample_count and cols >= 3:
        return data[:, :3]
    if cols == sample_count and rows >= 3:
        return data[:3, :].T
    if data.size == 3 * sample_count:
        return data.reshape(sample_count, 3)
    raise ValueError(f"Cannot interpret {name} shape {data.shape} for {sample_count} samples.")


def _sorted_trimmed_series(time, xyz, *, time_min: float, time_max: float):
    order = np.argsort(time)
    time = time[order]
    xyz = xyz[order]
    mask = (time >= time_min) & (time <= time_max)
    if not np.any(mask):
        raise ValueError(f"No samples are inside [{time_min}, {time_max}] s.")
    return time[mask], xyz[mask]


def _save_png_and_eps(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".eps"), format="eps")


def load_trajectories(mat_path: Path = DEFAULT_DATA_FILE) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = _load_data_verify(mat_path)
    series = {}
    for key in ("expm", "comsol", "ssm"):
        time = _as_time_vector(getattr(data, f"t_{key}"), f"t_{key}")
        xyz = _as_xyz(getattr(data, f"y_{key}"), time.size, f"y_{key}")
        series[key] = _sorted_trimmed_series(time, xyz, time_min=TIME_MIN, time_max=TIME_MAX)
    return series


def plot_xyz_time(series: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path, *, show: bool) -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    labels = {
        "expm": "Experiment",
        "comsol": "COMSOL full-order",
        "ssm": "SSM",
    }
    styles = {
        "expm": {"color": "#222222", "linestyle": "-", "linewidth": 2.0},
        "comsol": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.9},
        "ssm": {"color": "#d62728", "linestyle": "-.", "linewidth": 1.9},
    }
    axis_labels = ("X", "Y", "Z")

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.2), sharex=True, constrained_layout=True)
    handles = []
    legend_labels = []
    for axis_idx, ax in enumerate(axes):
        for key in ("expm", "comsol", "ssm"):
            time, xyz = series[key]
            line = ax.plot(time, xyz[:, axis_idx], label=labels[key], **styles[key])[0]
            if axis_idx == 0:
                handles.append(line)
                legend_labels.append(labels[key])
        ax.set_ylabel(f"{axis_labels[axis_idx]} [mm]")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(TIME_MIN, TIME_MAX)
    axes[-1].set_xticks(np.arange(TIME_MIN, TIME_MAX + 0.25, 0.5))
    fig.legend(
        handles,
        legend_labels,
        frameon=True,
        ncols=1,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        framealpha=0.72,
        facecolor="white",
        edgecolor="#888888",
    )
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _set_axes_equal_3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_3d_trajectory(series: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path, *, show: bool) -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    styles = {
        "expm": ("#222222", "-", "Experiment"),
        "comsol": ("#1f77b4", "--", "COMSOL full-order"),
        "ssm": ("#d62728", "-.", "SSM"),
    }
    for key in ("expm", "comsol", "ssm"):
        _, xyz = series[key]
        color, linestyle, label = styles[key]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linestyle=linestyle, linewidth=2.0, label=label)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.legend(
        frameon=True,
        framealpha=0.72,
        facecolor="white",
        edgecolor="#888888",
    )
    _set_axes_equal_3d(ax)
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    mat_path = Path(args.data_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_trajectories(mat_path)
    plot_xyz_time(series, output_dir / "data_verify_xyz_time.png", show=args.show)
    plot_3d_trajectory(series, output_dir / "data_verify_3d_trajectory.png", show=args.show)
    print(f"Saved figures to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DataVerify experiment, COMSOL, and SSM trajectories.")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="Path to DataVerify.mat.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated figures.")
    parser.add_argument("--show", action="store_true", help="Display figures in addition to saving them.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
