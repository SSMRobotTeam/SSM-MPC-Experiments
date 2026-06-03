from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ESTIMATE_CSV = SCRIPT_DIR / "results" / "dynamic_pk_estimates.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "ssm_estimate_plot"
AXES = ("x", "y", "z")
FONT_SIZE = 16
ENDEFF_INITIAL_POSITION_MM = np.asarray([0.0, 0.0, 440.0], dtype=float)


plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    }
)


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        columns: dict[str, list[float | str]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row[name]
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(value)

    return {
        name: np.asarray(values, dtype=float) if values and isinstance(values[0], float) else np.asarray(values)
        for name, values in columns.items()
    }


def _require_columns(data: dict[str, np.ndarray], columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise KeyError(f"{label} is missing required columns: {', '.join(missing)}")


def _time_vector(data: dict[str, np.ndarray], column: str) -> np.ndarray:
    _require_columns(data, [column], "SSM estimate CSV")
    return np.asarray(data[column], dtype=float)


def _xyz_matrix(data: dict[str, np.ndarray], prefix: str, *, flip_x: bool = False) -> np.ndarray:
    columns = [f"{prefix}_{axis}_mm" for axis in AXES]
    _require_columns(data, columns, "SSM estimate CSV")
    xyz = np.column_stack([np.asarray(data[column], dtype=float) for column in columns])
    if flip_x:
        xyz[:, 0] *= -1.0
    return xyz


def _six_column_matrix(data: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    columns = [f"{prefix}_{idx + 1}" for idx in range(6)]
    _require_columns(data, columns, "SSM estimate CSV")
    return np.column_stack([np.asarray(data[column], dtype=float) for column in columns])


def _motor_sent_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    columns = [f"motor_sent_u{idx + 1}_mm" for idx in range(6)]
    _require_columns(data, columns, "SSM estimate CSV")
    return np.column_stack([np.asarray(data[column], dtype=float) for column in columns])


def _save_png_and_eps(fig, output_path: Path, *, dpi: int = 260) -> None:
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".eps"), format="eps", dpi=dpi)


def _set_equal_3d_axes(ax, point_sets: list[np.ndarray]) -> None:
    points = np.vstack([points.reshape(-1, 3) for points in point_sets if points.size])
    center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    radius = 0.5 * float(np.max(np.ptp(points, axis=0)))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_endeff_time(
    time_s: np.ndarray,
    measured: np.ndarray,
    ssm: np.ndarray,
    output_path: Path,
    *,
    show: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.4), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.89, hspace=0.16)
    labels = ("X", "Y", "Z")
    legend_handles = []
    legend_labels = []
    for axis_idx, ax in enumerate(axes):
        measured_line = ax.plot(time_s, measured[:, axis_idx], color="black", linewidth=2.0, label="Measured")[0]
        ssm_line = ax.plot(
            time_s,
            ssm[:, axis_idx],
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label="SSM estimate",
        )[0]
        if axis_idx == 0:
            legend_handles.extend([measured_line, ssm_line])
            legend_labels.extend(["Measured", "SSM estimate"])
        ax.set_ylabel(f"{labels[axis_idx]} [mm]")
        ax.grid(True, color="#dddddd")
    axes[-1].set_xlabel("Time [s]")
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_endeff_3d(measured: np.ndarray, ssm: np.ndarray, output_path: Path, *, show: bool) -> None:
    measured_position = measured + ENDEFF_INITIAL_POSITION_MM.reshape(1, 3)
    ssm_position = ssm + ENDEFF_INITIAL_POSITION_MM.reshape(1, 3)
    fig = plt.figure(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.91)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=24, azim=-48)
    ax.plot(
        measured_position[:, 0],
        measured_position[:, 1],
        measured_position[:, 2],
        color="black",
        linewidth=2.0,
        label="Measured",
    )
    ax.plot(
        ssm_position[:, 0],
        ssm_position[:, 1],
        ssm_position[:, 2],
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="SSM estimate",
    )
    ax.set_xlabel("X [mm]", labelpad=8)
    ax.set_ylabel("Y [mm]", labelpad=8)
    ax.set_zlabel("Z [mm]", labelpad=8)
    ax.tick_params(axis="both", pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, color="#c7c7c7", linewidth=0.9)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.74, 0.74, 0.74, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.9
    _set_equal_3d_axes(ax, [measured_position, ssm_position])
    ax.set_box_aspect((1.0, 1.0, 1.15))
    ax.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cables(
    time_s: np.ndarray,
    cable_target: np.ndarray,
    cable_delta: np.ndarray,
    output_path: Path,
    *,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 6.8), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.84, wspace=0.24, hspace=0.24)
    axes = axes.reshape(-1)
    legend_handles = []
    legend_labels = []
    for cable_idx, ax in enumerate(axes):
        target_line = ax.plot(time_s, cable_target[:, cable_idx], color="black", linewidth=1.9, label="Target")[0]
        ssm_line = ax.plot(
            time_s,
            cable_delta[:, cable_idx],
            color="#d62728",
            linestyle="--",
            linewidth=1.9,
            label="SSM estimate",
        )[0]
        if cable_idx == 0:
            legend_handles.extend([target_line, ssm_line])
            legend_labels.extend(["Target", "SSM estimate"])
        ax.set_ylabel(f"Cable {cable_idx + 1} [mm]")
        ax.grid(True, color="#dddddd")
    for ax in axes[3:]:
        ax.set_xlabel("Time [s]")
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_cable_axes(
    axes: np.ndarray,
    time_s: np.ndarray,
    cable_target: np.ndarray,
    cable_delta: np.ndarray,
) -> tuple[list, list]:
    axes = axes.reshape(-1)
    legend_handles = []
    legend_labels = []
    for cable_idx, ax in enumerate(axes):
        target_line = ax.plot(time_s, cable_target[:, cable_idx], color="black", linewidth=1.9, label="Target")[0]
        ssm_line = ax.plot(
            time_s,
            cable_delta[:, cable_idx],
            color="#d62728",
            linestyle="--",
            linewidth=1.9,
            label="SSM estimate",
        )[0]
        if cable_idx == 0:
            legend_handles.extend([target_line, ssm_line])
            legend_labels.extend(["Target", "SSM estimate"])
        ax.set_ylabel(f"Cable {cable_idx + 1} [mm]")
        ax.grid(True, color="#dddddd")
    for ax in axes[3:]:
        ax.set_xlabel("Time [s]")
    return legend_handles, legend_labels


def _draw_endeff_3d_axis(ax, measured: np.ndarray, ssm: np.ndarray) -> tuple[list, list]:
    measured_position = measured + ENDEFF_INITIAL_POSITION_MM.reshape(1, 3)
    ssm_position = ssm + ENDEFF_INITIAL_POSITION_MM.reshape(1, 3)
    measured_line = ax.plot(
        measured_position[:, 0],
        measured_position[:, 1],
        measured_position[:, 2],
        color="black",
        linewidth=2.0,
        label="Measured",
    )[0]
    ssm_line = ax.plot(
        ssm_position[:, 0],
        ssm_position[:, 1],
        ssm_position[:, 2],
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="SSM estimate",
    )[0]
    start_marker = ax.scatter(
        measured_position[0, 0],
        measured_position[0, 1],
        measured_position[0, 2],
        color="black",
        s=45,
        marker="o",
        depthshade=False,
        label="Start",
        zorder=5,
    )
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("X [mm]", labelpad=8)
    ax.set_ylabel("Y [mm]", labelpad=8)
    ax.set_zlabel("Z [mm]", labelpad=8)
    ax.tick_params(axis="both", pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, color="#dddddd")
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1.0)
    _set_equal_3d_axes(ax, [measured_position, ssm_position])
    ax.set_box_aspect((1.2, 1.2, 1.056))
    return [measured_line, ssm_line, start_marker], ["Measured", "SSM estimate", "Start"]


def plot_cables_and_endeff_3d(
    time_s: np.ndarray,
    cable_target: np.ndarray,
    cable_delta: np.ndarray,
    measured: np.ndarray,
    ssm: np.ndarray,
    output_path: Path,
    *,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(18.0, 7.4))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.84, wspace=0.06, hspace=0.24)
    outer_grid = fig.add_gridspec(1, 2, width_ratios=(3.05, 2.28), wspace=-0.1)
    cable_grid = outer_grid[0, 0].subgridspec(2, 3, wspace=0.32, hspace=0.24)
    cable_axes = np.asarray([[fig.add_subplot(cable_grid[row, col]) for col in range(3)] for row in range(2)])
    ax_3d = fig.add_subplot(outer_grid[0, 1], projection="3d")

    cable_handles, _ = _draw_cable_axes(cable_axes, time_s, cable_target, cable_delta)
    trajectory_handles, _ = _draw_endeff_3d_axis(ax_3d, measured, ssm)

    cable_rmse = np.sqrt(np.mean((cable_delta - cable_target) ** 2, axis=0))
    position_error = (ssm + ENDEFF_INITIAL_POSITION_MM.reshape(1, 3)) - (
        measured + ENDEFF_INITIAL_POSITION_MM.reshape(1, 3)
    )
    endeff_rmse = float(np.sqrt(np.mean(np.sum(position_error**2, axis=1))))
    table_ax = fig.add_axes([0.055, 0.855, 0.60, 0.12])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=[["RMSE [mm]"] + [f"{value:.3f}" for value in cable_rmse] + [f"{endeff_rmse:.3f}"]],
        colLabels=["Metric"] + [f"C{idx + 1}" for idx in range(6)] + ["End"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)
        cell.set_facecolor("white")
        if row_idx == 0 or col_idx == 0:
            cell.set_text_props(weight="bold")
        if row_idx == 0:
            cell.visible_edges = "TB"
        elif row_idx == 1:
            cell.visible_edges = "B"

    fig.legend(
        cable_handles + [trajectory_handles[2]],
        ["Target", "SSM estimate", "Start"],
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.965),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_motor_commands(
    time_s: np.ndarray,
    motor_sent: np.ndarray,
    motor_command: np.ndarray,
    output_path: Path,
    *,
    show: bool,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8.6), sharex=True, constrained_layout=True)
    axes = axes.reshape(-1)
    for motor_idx, ax in enumerate(axes):
        ax.plot(time_s, motor_sent[:, motor_idx], color="black", linewidth=1.9, label="Sent")
        ax.plot(time_s, motor_command[:, motor_idx], color="#d62728", linestyle="--", linewidth=1.9, label="SSM command")
        ax.set_ylabel(f"Motor {motor_idx + 1} [mm]")
        ax.grid(True, color="#dddddd")
        ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0, loc="best")
    axes[-2].set_xlabel("Time [s]")
    axes[-1].set_xlabel("Time [s]")
    _save_png_and_eps(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_summary_csv(
    measured: np.ndarray,
    ssm: np.ndarray,
    cable_target: np.ndarray,
    cable_delta: np.ndarray,
    motor_sent: np.ndarray,
    motor_command: np.ndarray,
    output_path: Path,
) -> None:
    endeff_error = ssm - measured
    cable_error = cable_delta - cable_target
    motor_error = motor_command - motor_sent
    cable_rmse = np.sqrt(np.mean(cable_error**2, axis=0))
    rows = [
        ("endeff_xyz_rms_mm", float(np.sqrt(np.mean(endeff_error**2)))),
        ("endeff_xyz_rmse_norm_mm", float(np.sqrt(np.mean(np.sum(endeff_error**2, axis=1))))),
        ("cable_target_vs_delta_rms_mm", float(np.sqrt(np.mean(cable_error**2)))),
        ("motor_sent_vs_command_rms_mm", float(np.sqrt(np.mean(motor_error**2)))),
    ]
    rows.extend((f"cable_{idx + 1}_ssm_estimate_rmse_mm", float(value)) for idx, value in enumerate(cable_rmse))
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "value"))
        writer.writerows((name, f"{value:.8f}") for name, value in rows)


def write_cable_rmse_csv(cable_target: np.ndarray, cable_delta: np.ndarray, output_path: Path) -> np.ndarray:
    cable_error = cable_delta - cable_target
    cable_rmse = np.sqrt(np.mean(cable_error**2, axis=0))
    overall_rmse = float(np.sqrt(np.mean(cable_error**2)))
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("cable", "rmse_mm"))
        for idx, value in enumerate(cable_rmse):
            writer.writerow((idx + 1, f"{value:.8f}"))
        writer.writerow(("overall", f"{overall_rmse:.8f}"))
    return cable_rmse


def run(args: argparse.Namespace) -> None:
    estimate_csv = Path(args.estimate_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_csv(estimate_csv)
    time_s = _time_vector(data, args.time_column)
    measured_endeff = _xyz_matrix(data, "measurement_endeff", flip_x=True)
    ssm_endeff = _xyz_matrix(data, "ssm_endeff_displacement")
    cable_target = _six_column_matrix(data, "cable_target_mm")
    cable_delta = _six_column_matrix(data, "cable_delta_mm")
    motor_sent = _motor_sent_matrix(data)
    motor_command = _six_column_matrix(data, "motor_command_mm")

    plot_endeff_time(time_s, measured_endeff, ssm_endeff, output_dir / "ssm_estimate_endeff_time.png", show=args.show)
    plot_endeff_3d(measured_endeff, ssm_endeff, output_dir / "ssm_estimate_endeff_3d.png", show=args.show)
    plot_cables(time_s, cable_target, cable_delta, output_dir / "ssm_estimate_cable_controls.png", show=args.show)
    plot_cables_and_endeff_3d(
        time_s,
        cable_target,
        cable_delta,
        measured_endeff,
        ssm_endeff,
        output_dir / "ssm_estimate_cables_and_endeff_3d.png",
        show=args.show,
    )
    plot_motor_commands(time_s, motor_sent, motor_command, output_dir / "ssm_estimate_motor_commands.png", show=args.show)
    write_summary_csv(
        measured_endeff,
        ssm_endeff,
        cable_target,
        cable_delta,
        motor_sent,
        motor_command,
        output_dir / "ssm_estimate_summary.csv",
    )
    cable_rmse = write_cable_rmse_csv(cable_target, cable_delta, output_dir / "ssm_estimate_cable_rmse.csv")
    print(f"Saved SSM estimate figures and summary to {output_dir}")
    print("SSM estimate cable RMSE [mm]: " + ", ".join(f"cable {idx + 1}={value:.6f}" for idx, value in enumerate(cable_rmse)))
    print(f"Overall cable RMSE [mm]: {float(np.sqrt(np.mean((cable_delta - cable_target) ** 2))):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SSM estimate end-effector trajectory and cable controls.")
    parser.add_argument("--estimate-csv", type=Path, default=DEFAULT_ESTIMATE_CSV, help="dynamic_pk_estimates.csv path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output figure folder.")
    parser.add_argument(
        "--time-column",
        default="actual_time_s",
        choices=("actual_time_s", "planned_time_s"),
        help="Time column used on x-axis.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
