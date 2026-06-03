from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "Diamond_DATA_comp_SUT_Trajs"
OUTPUT_DIR = SCRIPT_DIR / "figures"

TRAJECTORIES = {
    "S": DATA_ROOT / "0806-comp-S-symbol",
    "U": DATA_ROOT / "0806-comp-U-symbol",
    "T": DATA_ROOT / "0806-comp-T-symbol",
}
MODEL_DIRS = {
    "SSM": "data_SSM",
    "POD26": "data_POD26",
    "POD6": "data_POD6",
}
DISPLAY_LABELS = {
    "SSM": "SSM(6)",
    "POD26": "POD(26)",
    "POD6": "POD(6)",
}
MODEL_STYLES = {
    "Desired": {"color": "k", "linestyle": "--", "linewidth": 3.0},
    "SSM": {"color": "tab:blue", "linestyle": "-", "linewidth": 3.0},
    "POD26": {"color": "tab:orange", "linestyle": "-", "linewidth": 3.0},
    "POD6": {"color": "tab:green", "linestyle": "-", "linewidth": 3.0},
}
FONT_SIZE = 20


def _load_mat_variable(path: Path, variable_name: str) -> np.ndarray:
    data = sciIO.loadmat(str(path), squeeze_me=True)
    if variable_name not in data:
        available = ", ".join(key for key in data if not key.startswith("__"))
        raise KeyError(f"Missing variable '{variable_name}' in {path}. Available: {available}")
    return np.asarray(data[variable_name], dtype=float)


def _as_n_by_3(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D trajectory array, got shape {values.shape}.")
    if values.shape[0] == 3 and values.shape[1] != 3:
        return values.T
    if values.shape[1] >= 3:
        return values[:, :3]
    if values.shape[0] >= 3:
        return values[:3, :].T
    raise ValueError(f"Expected xyz rows or columns, got shape {values.shape}.")


def load_model_trajectory(trajectory_dir: Path, model_dir_name: str) -> dict:
    model_dir = trajectory_dir / model_dir_name
    t = _load_mat_variable(model_dir / "t.mat", "t").reshape(-1)
    desired = _as_n_by_3(_load_mat_variable(model_dir / "zzd.mat", "zzd"))
    measured = _as_n_by_3(_load_mat_variable(model_dir / "y_exprm.mat", "y_exprm"))

    sample_count = min(t.size, desired.shape[0], measured.shape[0])
    return {
        "t": t[:sample_count],
        "desired": desired[:sample_count],
        "measured": measured[:sample_count],
    }


def load_trajectory(symbol: str) -> dict:
    symbol = symbol.upper()
    if symbol not in TRAJECTORIES:
        choices = ", ".join(TRAJECTORIES)
        raise ValueError(f"Unknown trajectory '{symbol}'. Choose one of: {choices}")

    trajectory_dir = TRAJECTORIES[symbol]
    models = {
        label: load_model_trajectory(trajectory_dir, model_dir)
        for label, model_dir in MODEL_DIRS.items()
    }
    desired = models["SSM"]["desired"]
    sample_count = min([desired.shape[0], *[model["measured"].shape[0] for model in models.values()]])
    return {
        "symbol": symbol,
        "desired": desired[:sample_count],
        "models": {
            label: {
                "t": model["t"][:sample_count],
                "measured": model["measured"][:sample_count],
            }
            for label, model in models.items()
        },
    }


def rmse_xy(measured: np.ndarray, desired: np.ndarray) -> float:
    residual = np.asarray(measured, dtype=float)[:, :2] - np.asarray(desired, dtype=float)[:, :2]
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def _set_equal_xy(ax, point_sets: list[np.ndarray], padding_ratio: float = 0.08) -> None:
    points = np.vstack([np.asarray(points, dtype=float)[:, :2] for points in point_sets if points.size])
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.max(upper - lower))
    radius = max(radius, 1.0)
    radius *= 1.0 + padding_ratio
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_sut_trajectories(output_dir: Path = OUTPUT_DIR, show: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_trajectory(symbol) for symbol in ("S", "U", "T")]

    fig = plt.figure(figsize=(22.0, 18.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.035, wspace=0.035, hspace=0.02)
    grid = fig.add_gridspec(6, 4, width_ratios=(1.0, 1.0, 1.0, 1.0))
    for group_idx, dataset in enumerate(datasets):
        symbol = dataset["symbol"]
        desired = dataset["desired"]
        row_start = 2 * group_idx
        ax_rmse = fig.add_subplot(grid[row_start : row_start + 2, 3])

        model_labels = []
        model_rmse = []
        model_colors = []
        for model_col, (model_label, model) in enumerate(dataset["models"].items()):
            ax_traj = fig.add_subplot(grid[row_start : row_start + 2, model_col])
            measured = model["measured"]
            rmse_value = rmse_xy(measured, desired)
            model_labels.append(DISPLAY_LABELS[model_label])
            model_rmse.append(rmse_value)
            model_colors.append(MODEL_STYLES[model_label]["color"])

            ax_traj.plot(
                desired[:, 0],
                desired[:, 1],
                label="Desired",
                **MODEL_STYLES["Desired"],
            )
            ax_traj.plot(
                measured[:, 0],
                measured[:, 1],
                label=DISPLAY_LABELS[model_label],
                **MODEL_STYLES[model_label],
            )
            ax_traj.set_title(DISPLAY_LABELS[model_label], fontsize=FONT_SIZE)
            ax_traj.set_xlabel("X [mm]", fontsize=FONT_SIZE)
            ax_traj.set_ylabel("Y [mm]", fontsize=FONT_SIZE)
            ax_traj.tick_params(labelsize=FONT_SIZE)
            ax_traj.grid(True, alpha=0.3)
            _set_equal_xy(ax_traj, [desired, measured])

        bars = ax_rmse.bar(model_labels, model_rmse, color=model_colors, width=0.58)
        max_rmse = max(model_rmse) if model_rmse else 1.0
        ax_rmse.set_ylim(0.0, max_rmse * 1.15)
        for bar, rmse_value in zip(bars, model_rmse):
            ax_rmse.annotate(
                f"{rmse_value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
            )
        ax_rmse.set_ylabel("RMSE [mm]", fontsize=FONT_SIZE)
        ax_rmse.tick_params(labelsize=FONT_SIZE)
        ax_rmse.grid(True, axis="y", alpha=0.3)
        ax_rmse.set_box_aspect(1)

    output_path = output_dir / "diamond_sut_trajectories_6x2_rmse.png"
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".eps"), format="eps", dpi=240)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Diamond S/U/T trajectory comparisons with RMSE.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for output figures.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved_path = plot_sut_trajectories(output_dir=args.output_dir, show=args.show)
    print(f"Saved {saved_path}")
