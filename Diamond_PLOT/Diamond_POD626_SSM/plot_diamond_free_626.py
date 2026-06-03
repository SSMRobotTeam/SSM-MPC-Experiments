from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"
DEFAULT_FIG_FILE = DATA_DIR / "diamond_free_626.fig"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"

DISPLAY_LABELS = {
    "Comsol": "COMSOL",
    "SSM $\\mathcal{O}$(6)": "SSM(6)",
    "POD $\\mathcal{O}$(6)": "POD(6)",
    "POD $\\mathcal{O}$(26)": "POD(26)",
}
STYLE_MAP = {
    "COMSOL": {"color": "black", "linestyle": "-", "linewidth": 2.4},
    "SSM(6)": {"color": "tab:blue", "linestyle": "--", "linewidth": 2.2},
    "POD(6)": {"color": "tab:green", "linestyle": "-.", "linewidth": 2.2},
    "POD(26)": {"color": "magenta", "linestyle": ":", "linewidth": 2.4},
}
AXIS_LABELS = ("X", "Y", "Z")
FONT_SIZE = 16


def _children(obj) -> list:
    raw_children = getattr(obj, "children", [])
    children = np.asarray(raw_children, dtype=object)
    if children.size == 0:
        return []
    return list(children.reshape(-1))


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    array = np.asarray(value, dtype=object)
    if array.shape == ():
        item = array.item()
        return item if isinstance(item, str) else str(item)
    return " ".join(str(item) for item in array.reshape(-1))


def _line_name(line_obj, fallback_index: int) -> str:
    properties = line_obj.properties
    raw_name = getattr(properties, "DisplayName", "")
    label = DISPLAY_LABELS.get(_as_text(raw_name), _as_text(raw_name))
    if label:
        return label
    ordered_labels = ("COMSOL", "SSM(6)", "POD(6)", "POD(26)")
    return ordered_labels[fallback_index]


def _axis_label_from_texts(axis_obj) -> str | None:
    for child in _children(axis_obj):
        if getattr(child, "type", "") != "text":
            continue
        text = _as_text(getattr(child.properties, "String", ""))
        for axis_label in AXIS_LABELS:
            if f"{axis_label}/" in text:
                return axis_label
    return None


def _main_axes_from_fig(fig_file: Path) -> list:
    mat_data = sciIO.loadmat(str(fig_file), squeeze_me=True, struct_as_record=False)
    if "hgS_070000" not in mat_data:
        raise KeyError(f"Cannot find MATLAB figure root 'hgS_070000' in {fig_file}.")

    axes = []
    for child in _children(mat_data["hgS_070000"]):
        if getattr(child, "type", "") != "axes":
            continue
        axis_label = _axis_label_from_texts(child)
        lines = [line for line in _children(child) if "lineseries" in getattr(line, "type", "")]
        if axis_label in AXIS_LABELS and len(lines) >= 4:
            axes.append((axis_label, child, lines[:4]))

    axes.sort(key=lambda item: AXIS_LABELS.index(item[0]))
    if [axis_label for axis_label, _, _ in axes] != list(AXIS_LABELS):
        found = ", ".join(axis_label for axis_label, _, _ in axes)
        raise ValueError(f"Expected main X/Y/Z axes in {fig_file}; found: {found}")
    return axes


def load_trajectories(fig_file: Path = DEFAULT_FIG_FILE) -> dict[str, dict[str, np.ndarray]]:
    series: dict[str, dict[str, list[np.ndarray] | np.ndarray]] = {}
    for axis_label, _, lines in _main_axes_from_fig(fig_file):
        axis_idx = AXIS_LABELS.index(axis_label)
        for line_index, line in enumerate(lines):
            label = _line_name(line, line_index)
            properties = line.properties
            time = np.asarray(properties.XData, dtype=float).reshape(-1)
            values = np.asarray(properties.YData, dtype=float).reshape(-1)
            if label not in series:
                series[label] = {"time": [], "values": [None, None, None]}
            series[label]["time"].append(time)
            series[label]["values"][axis_idx] = values

    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for label, raw in series.items():
        times = raw["time"]
        values_by_axis = raw["values"]
        if any(values is None for values in values_by_axis):
            raise ValueError(f"Trajectory '{label}' does not have complete X/Y/Z data.")
        if not all(np.array_equal(times[0], time) for time in times[1:]):
            # Keep each model on its native time grid by interpolating Y/Z onto X time.
            base_time = np.asarray(times[0], dtype=float)
            values = np.column_stack(
                [
                    np.interp(base_time, np.asarray(times[idx], dtype=float), np.asarray(values_by_axis[idx], dtype=float))
                    for idx in range(3)
                ]
            )
        else:
            base_time = np.asarray(times[0], dtype=float)
            values = np.column_stack([np.asarray(values_by_axis[idx], dtype=float) for idx in range(3)])
        trajectories[label] = {"time": base_time, "xyz": values}

    expected = ("COMSOL", "SSM(6)", "POD(6)", "POD(26)")
    missing = [label for label in expected if label not in trajectories]
    if missing:
        raise KeyError(f"Missing trajectories in {fig_file}: {', '.join(missing)}")
    return {label: trajectories[label] for label in expected}


def compute_rmse_to_comsol(trajectories: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    comsol = trajectories["COMSOL"]
    rmse_values = {}
    for label in ("SSM(6)", "POD(6)", "POD(26)"):
        model = trajectories[label]
        t_start = max(float(comsol["time"][0]), float(model["time"][0]))
        t_end = min(float(comsol["time"][-1]), float(model["time"][-1]))
        mask = (comsol["time"] >= t_start) & (comsol["time"] <= t_end)
        common_time = comsol["time"][mask]
        if common_time.size < 2:
            raise ValueError(f"Not enough overlapping samples to compute RMSE for {label}.")
        model_xyz = np.column_stack(
            [
                np.interp(common_time, model["time"], model["xyz"][:, axis_idx])
                for axis_idx in range(3)
            ]
        )
        residual = model_xyz - comsol["xyz"][mask, :]
        rmse_values[label] = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return rmse_values


def _save_png_and_eps(fig, output_path: Path, *, dpi: int = 260) -> None:
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".eps"), format="eps", dpi=dpi)


def write_rmse_csv(rmse_values: dict[str, float], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "rmse_xyz_mm"))
        writer.writeheader()
        for label, value in rmse_values.items():
            writer.writerow({"model": label, "rmse_xyz_mm": f"{value:.8f}"})


def plot_trajectories(
    trajectories: dict[str, dict[str, np.ndarray]],
    rmse_values: dict[str, float],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    show: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 10.0))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.08, top=0.92, hspace=0.42)
    grid = fig.add_gridspec(4, 1, height_ratios=(1.0, 1.0, 1.0, 0.62))
    axes = [fig.add_subplot(grid[idx, 0]) for idx in range(3)]
    table_ax = fig.add_subplot(grid[3, 0])

    for axis_idx, ax in enumerate(axes):
        for label, dataset in trajectories.items():
            ax.plot(dataset["time"], dataset["xyz"][:, axis_idx], label=label, **STYLE_MAP[label])
        ax.set_ylabel(f"{AXIS_LABELS[axis_idx]} [mm]", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.grid(True, color="#dddddd")
        ax.set_xlim(0.0, 0.5)
    axes[-1].set_xlabel("Time [s]", fontsize=FONT_SIZE)
    axes[0].legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.38),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fontsize=FONT_SIZE,
    )

    table_ax.axis("off")
    models = list(rmse_values)
    table = table_ax.table(
        cellText=[["RMSE XYZ [mm]"] + [f"{rmse_values[label]:.3f}" for label in models]],
        colLabels=["Metric"] + models,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE)
    table.scale(1.0, 2.0)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2)
        cell.set_facecolor("white")
        if row_idx == 0 or col_idx == 0:
            cell.set_text_props(weight="bold")
        if row_idx == 0:
            cell.visible_edges = "TB"
        elif row_idx == 1:
            cell.visible_edges = "B"

    output_path = output_dir / "diamond_free_626_trajectory_rmse.png"
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def run(args: argparse.Namespace) -> None:
    fig_file = Path(args.fig_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    trajectories = load_trajectories(fig_file)
    rmse_values = compute_rmse_to_comsol(trajectories)
    output_path = plot_trajectories(trajectories, rmse_values, output_dir, show=bool(args.show))
    write_rmse_csv(rmse_values, output_dir / "diamond_free_626_rmse.csv")
    print(f"Saved {output_path}")
    for label, value in rmse_values.items():
        print(f"{label} RMSE XYZ = {value:.6f} mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Diamond free trajectories from diamond_free_626.fig and compute RMSE.")
    parser.add_argument("--fig-file", type=Path, default=DEFAULT_FIG_FILE, help="Path to diamond_free_626.fig.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for output figures and RMSE CSV.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
