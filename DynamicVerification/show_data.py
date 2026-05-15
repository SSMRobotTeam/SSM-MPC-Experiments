from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MARKER_NAMES = ("endeff", "mideff", "checkpoint")
AXES = ("x", "y", "z")
AXIS_COLORS = {
    "x": "#d62728",
    "y": "#2ca02c",
    "z": "#1f77b4",
}
MARKER_COLORS = {
    "endeff": "#d62728",
    "mideff": "#1f77b4",
    "checkpoint": "#2ca02c",
}


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

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


def _time_vector(data: dict[str, np.ndarray], preferred: str) -> np.ndarray:
    if preferred not in data:
        raise KeyError(f"Missing time column: {preferred}")
    return np.asarray(data[preferred], dtype=float)


def _require_columns(data: dict[str, np.ndarray], names: list[str]) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


def plot_motor_controls(time_s: np.ndarray, data: dict[str, np.ndarray], output_path: Path) -> plt.Figure:
    motor_columns = [f"motor_sent_u{idx}_mm" for idx in range(1, 7)]
    _require_columns(data, motor_columns)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for idx, column in enumerate(motor_columns, start=1):
        ax.plot(time_s, data[column], linewidth=1.8, label=f"u{idx}")

    ax.set_title("Motor Control Inputs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor command (mm)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(ncol=3)
    fig.savefig(output_path, dpi=300)
    return fig


def plot_marker_displacement(
    marker_name: str,
    time_s: np.ndarray,
    data: dict[str, np.ndarray],
    output_path: Path,
) -> plt.Figure:
    columns = [f"{marker_name}_disp_{axis}_ref_mm" for axis in AXES]
    _require_columns(data, columns)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for axis, column in zip(AXES, columns):
        ax.plot(
            time_s,
            data[column],
            color=AXIS_COLORS[axis],
            linewidth=1.9,
            label=f"{axis.upper()} displacement",
        )

    ax.set_title(f"{marker_name} Displacement Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    fig.savefig(output_path, dpi=300)
    return fig


def plot_3d_trajectories(data: dict[str, np.ndarray], output_path: Path) -> plt.Figure:
    columns = [
        f"{marker_name}_disp_{axis}_ref_mm"
        for marker_name in MARKER_NAMES
        for axis in AXES
    ]
    _require_columns(data, columns)

    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(0.0, 0.0, 0.0, color="black", marker="x", s=70, linewidths=2.2, label="origin")
    ax.text(0.0, 0.0, 0.0, "  (0, 0, 0)", color="black")

    for marker_name in MARKER_NAMES:
        x = data[f"{marker_name}_disp_x_ref_mm"]
        y = data[f"{marker_name}_disp_y_ref_mm"]
        z = data[f"{marker_name}_disp_z_ref_mm"]
        color = MARKER_COLORS[marker_name]

        ax.plot(x, y, z, color=color, linewidth=2.0, label=marker_name)
        ax.scatter(x[0], y[0], z[0], color=color, marker="o", s=36)
        ax.scatter(x[-1], y[-1], z[-1], color=color, marker="^", s=52)

    ax.set_title("3D Motion Trajectories")
    ax.set_xlabel("X displacement (mm)")
    ax.set_ylabel("Y displacement (mm)")
    ax.set_zlabel("Z displacement (mm)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=300)
    return fig


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot dynamic verification data from DynData/Dyndata01.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "DynData" / "Dyndata01.csv",
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "show_data",
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--time-column",
        default="actual_time_s",
        choices=("actual_time_s", "planned_time_s"),
        help="Time column used as the x axis.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_csv(args.input)
    time_s = _time_vector(data, args.time_column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.input.stem

    figures = []
    motor_path = args.output_dir / f"{output_stem}_motor_controls.png"
    figures.append(plot_motor_controls(time_s, data, motor_path))
    print(f"Saved {motor_path}")

    for marker_name in MARKER_NAMES:
        output_path = args.output_dir / f"{output_stem}_{marker_name}_displacement.png"
        figures.append(plot_marker_displacement(marker_name, time_s, data, output_path))
        print(f"Saved {output_path}")

    trajectory_path = args.output_dir / f"{output_stem}_3d_trajectories.png"
    figures.append(plot_3d_trajectories(data, trajectory_path))
    print(f"Saved {trajectory_path}")

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
