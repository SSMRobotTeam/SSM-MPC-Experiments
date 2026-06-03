from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "figures"
FONT_SIZE = 16

DATASETS = {
    "SSM(6)": {
        "t": ("t_ssm.mat", "t"),
        "y": ("y_exprm_ssm.mat", "y_exprm"),
        "zd": ("zzd_ssm.mat", "zzd"),
        "color": "tab:blue",
        "linestyle": "-",
    },
    "POD(26)": {
        "t": ("t_pod26.mat", "t"),
        "y": ("y_exprm_pod26.mat", "y_exprm"),
        "zd": ("zzd_pod26.mat", "zzd"),
        "color": "tab:orange",
        "linestyle": "--",
        "prepend_zero_z": True,
    },
    "POD(6)": {
        "t": ("t_pod6.mat", "t_pod6"),
        "y": ("y_exprm_pod6.mat", "y_exprm_pod6"),
        "zd": ("zzd_pod6.mat", "zzd_pod6"),
        "color": "tab:green",
        "linestyle": "-.",
    },
}


def _load_mat_variable(file_name: str, variable_name: str) -> np.ndarray:
    path = SCRIPT_DIR / file_name
    data = sciIO.loadmat(str(path), squeeze_me=True)
    if variable_name not in data:
        available = ", ".join(key for key in data if not key.startswith("__"))
        raise KeyError(f"Missing variable '{variable_name}' in {path}. Available: {available}")
    return np.asarray(data[variable_name], dtype=float)


def _as_n_by_3(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected 2-D data, got {values.shape}.")
    if values.shape[0] == 3 and values.shape[1] != 3:
        return values.T
    if values.shape[1] >= 3:
        return values[:, :3]
    if values.shape[0] >= 3:
        return values[:3, :].T
    raise ValueError(f"Expected xyz rows or columns, got {values.shape}.")


def load_dataset(label: str) -> dict:
    config = DATASETS[label]
    t = _load_mat_variable(*config["t"]).reshape(-1)
    measured = _as_n_by_3(_load_mat_variable(*config["y"]))
    desired = _as_n_by_3(_load_mat_variable(*config["zd"]))
    sample_count = min(t.size, desired.shape[0])
    if measured.shape[0] >= sample_count:
        measured = measured[-sample_count:]
    else:
        sample_count = measured.shape[0]
    return {
        "label": label,
        "color": config["color"],
        "linestyle": config["linestyle"],
        "prepend_zero_z": config.get("prepend_zero_z", False),
        "t": t[:sample_count],
        "measured": measured,
        "desired": desired[:sample_count],
    }


def rmse_xyz(measured: np.ndarray, desired: np.ndarray) -> float:
    residual = np.asarray(measured)[:, :3] - np.asarray(desired)[:, :3]
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def error_norm_xy(measured: np.ndarray, desired: np.ndarray) -> np.ndarray:
    residual = np.asarray(measured)[:, :2] - np.asarray(desired)[:, :2]
    return np.linalg.norm(residual, axis=1)


def _set_equal_xy(ax, point_sets: list[np.ndarray], padding_ratio: float = 0.08) -> None:
    points = np.vstack([np.asarray(points)[:, :2] for points in point_sets if points.size])
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.max(upper - lower))
    radius = max(radius, 1.0) * (1.0 + padding_ratio)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def _style_axis(ax) -> None:
    ax.tick_params(labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)


def plot_xy_tracking(datasets: list[dict], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    desired = datasets[0]["desired"]
    ax.plot(desired[:, 0], desired[:, 1], "k--", linewidth=2.2, label="Desired")
    point_sets = [desired]
    for dataset in datasets:
        measured = dataset["measured"]
        point_sets.append(measured)
        ax.plot(
            measured[:, 0],
            measured[:, 1],
            color=dataset["color"],
            linestyle=dataset["linestyle"],
            linewidth=4.0,
            label=f"{dataset['label']} RMSE={rmse_xyz(measured, dataset['desired']):.3f} mm",
        )
    ax.set_xlabel("X [mm]", fontsize=FONT_SIZE)
    ax.set_ylabel("Y [mm]", fontsize=FONT_SIZE)
    ax.set_title("Diamond Spiral XY Tracking", fontsize=FONT_SIZE)
    ax.legend(frameon=False, fontsize=FONT_SIZE)
    _style_axis(ax)
    _set_equal_xy(ax, point_sets)
    output_path = output_dir / "diamond_spiral_xy_tracking.png"
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".eps"), format="eps", dpi=240)
    plt.close(fig)
    return output_path


def plot_time_tracking(datasets: list[dict], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    flat_axes = axes.ravel()
    channels = ((0, "X"), (1, "Y"), (2, "Z"))
    reference = datasets[0]
    time_min = float(reference["t"][0])
    time_max = float(reference["t"][-1])
    for ax, (axis_idx, axis_name) in zip(flat_axes[:3], channels):
        ax.plot(reference["t"], reference["desired"][:, axis_idx], "k--", linewidth=2.2, label="Desired")
        for dataset in datasets:
            plot_t = dataset["t"]
            plot_values = dataset["measured"][:, axis_idx]
            if axis_idx == 2 and dataset["prepend_zero_z"]:
                plot_t = np.insert(plot_t, 0, 0.0)
                plot_values = np.insert(plot_values, 0, 0.0)
            ax.plot(
                plot_t,
                plot_values,
                color=dataset["color"],
                linestyle=dataset["linestyle"],
                linewidth=2.2,
                label=dataset["label"],
            )
        ax.set_ylabel(f"{axis_name} [mm]", fontsize=FONT_SIZE)
        ax.set_xlabel("Time [s]", fontsize=FONT_SIZE)
        ax.set_xlim(time_min, time_max)
        _style_axis(ax)
    flat_axes[0].legend(frameon=False, fontsize=FONT_SIZE, ncol=2)

    ax_rmse = flat_axes[3]
    labels = [dataset["label"] for dataset in datasets]
    colors = [dataset["color"] for dataset in datasets]
    rmse_values = [rmse_xyz(dataset["measured"], dataset["desired"]) for dataset in datasets]
    bars = ax_rmse.bar(labels, rmse_values, color=colors, width=0.58)
    ax_rmse.set_ylabel("RMSE [mm]", fontsize=FONT_SIZE)
    ax_rmse.set_ylim(0.0, max(rmse_values) * 1.18)
    for bar, value in zip(bars, rmse_values):
        ax_rmse.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
        )
    ax_rmse.set_title("RMSE", fontsize=FONT_SIZE)
    _style_axis(ax_rmse)

    output_path = output_dir / "diamond_spiral_xyz_rmse_2x2.png"
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".eps"), format="eps", dpi=240)
    plt.close(fig)
    return output_path


def plot_all(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(label) for label in DATASETS]
    saved_paths = [
        plot_xy_tracking(datasets, output_dir),
        plot_time_tracking(datasets, output_dir),
    ]
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Diamond Spiral tracking results.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for output figures.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for path in plot_all(args.output_dir):
        print(f"Saved {path}")
