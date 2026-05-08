from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.quat_rotR import quat2dcm
from dynamics.SSM.matlab_data import ROM_DIR, load_rom_data


TARGETS = ("mideff", "checkpoint", "Marker100152")
MEASUREMENT_TARGETS = ("Marker100152", "mideff", "checkpoint")
MEASUREMENT_NAMES = ("endeff", "mideff", "checkpoint")
AXES = ("X", "Y", "Z")
POS_END = np.array([0.0, 0.0, 0.4400])


def _metadata_pairs(row: list[str]) -> dict[str, str]:
    return {
        row[i]: row[i + 1]
        for i in range(0, len(row) - 1, 2)
        if row[i] and row[i + 1]
    }


def _column_index(
    names: list[str],
    measures: list[str],
    axes: list[str],
    object_name: str,
    measure: str,
    axis: str,
) -> int:
    for idx, (name, item_measure, item_axis) in enumerate(zip(names, measures, axes)):
        if name == object_name and item_measure == measure and item_axis == axis:
            return idx
    raise KeyError(f"Missing column: {object_name} {measure} {axis}")


def _read_motion_csv(path: Path) -> tuple[dict[str, str], list[str], list[str], list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = [next(reader) for _ in range(7)]
        data_rows = [row for row in reader if row and row[0]]

    metadata = _metadata_pairs(header[0])
    names = header[3]
    measures = header[5]
    axes = header[6]
    data = np.array(
        [[float(value) if value else np.nan for value in row] for row in data_rows],
        dtype=float,
    )
    return metadata, names, measures, axes, data


def _position(data: np.ndarray, names: list[str], measures: list[str], axes: list[str], object_name: str) -> np.ndarray:
    columns = [_column_index(names, measures, axes, object_name, "Position", axis) for axis in AXES]
    return data[:, columns]


def _base_rotation(data: np.ndarray, names: list[str], measures: list[str], axes: list[str]) -> np.ndarray:
    columns = [_column_index(names, measures, axes, "Base", "Rotation", axis) for axis in ("W", "X", "Y", "Z")]
    return data[:, columns]


def _local_position(point_xyz: np.ndarray, base_xyz: np.ndarray, base_to_global: np.ndarray) -> np.ndarray:
    return (base_to_global @ (point_xyz - base_xyz).T).T


def transform_to_reference_displacement(
    csv_path: Path,
    *,
    base_y_offset: float = -12.0,
) -> tuple[dict[str, str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata, names, measures, axes, data = _read_motion_csv(csv_path)
    time = data[:, 1]
    base_xyz = _position(data, names, measures, axes, "Base")[0].copy()
    base_xyz[1] -= base_y_offset
    base_q = _base_rotation(data, names, measures, axes)[0]
    base_to_global = quat2dcm(base_q)

    displacements: dict[str, np.ndarray] = {}
    initial_positions: dict[str, np.ndarray] = {}
    for target in TARGETS:
        target_position = _position(data, names, measures, axes, target)
        reference_position = _local_position(target_position, base_xyz, base_to_global)
        initial_positions[target] = reference_position[0]
        displacements[target] = reference_position - reference_position[0]
    return metadata, time, displacements, initial_positions


def save_processed_csv(output_path: Path, time: np.ndarray, displacements: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["time"]
    for target in TARGETS:
        header.extend([f"{target}_d{axis}_reference_mm" for axis in AXES])

    rows = []
    for idx, t in enumerate(time):
        row = [t]
        for target in TARGETS:
            row.extend(displacements[target][idx])
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot_time_curves(output_path: Path, source_name: str, time: np.ndarray, displacements: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    colors = {
        "mideff": "#1f77b4",
        "checkpoint": "#2ca02c",
        "Marker100152": "#d62728",
    }

    for axis_idx, axis_name in enumerate(AXES):
        ax = axes[axis_idx]
        for target in TARGETS:
            ax.plot(time, displacements[target][:, axis_idx], label=target, linewidth=1.2, color=colors[target])
        ax.set_ylabel(f"d{axis_name}_Ref (mm)")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best", ncols=3)
    fig.suptitle(f"{source_name}: displacement from initial Reference position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def extract_max_displacement_sample(
    source_name: str,
    time: np.ndarray,
    displacements: dict[str, np.ndarray],
    initial_positions: dict[str, np.ndarray],
    measurement_dofs_1based: np.ndarray | None = None,
    feedback_node_info: dict[str, object] | None = None,
) -> dict[str, object]:
    measurement_parts = []
    per_target = {}
    for measurement_name, target in zip(MEASUREMENT_NAMES, MEASUREMENT_TARGETS):
        norms = np.linalg.norm(displacements[target], axis=1)
        max_index = int(np.nanargmax(norms))
        max_vector = displacements[target][max_index]
        measurement_parts.append(max_vector)
        per_target[measurement_name] = {
            "source_target": target,
            "time_s": float(time[max_index]),
            "frame_index": max_index,
            "max_norm_mm": float(norms[max_index]),
            "displacement_mm": max_vector,
            "initial_position_ref_mm": initial_positions[target],
        }

    return {
        "source": source_name,
        "measurement_order": MEASUREMENT_NAMES,
        "measurement_mm": np.concatenate(measurement_parts),
        "per_target": per_target,
        "reference_initial_positions_mm": {
            "endeff": initial_positions["Marker100152"],
            "mideff": initial_positions["mideff"],
            "checkpoint": initial_positions["checkpoint"],
        },
        "measurement_dofs_1based": measurement_dofs_1based,
        "feedback_node_info": feedback_node_info,
    }


def static_feedback_node_info(initial_positions: dict[str, np.ndarray]) -> dict[str, object]:
    rom_data = load_rom_data(ROM_DIR)
    outdof_endeff = np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)
    outdof_half = np.asarray(rom_data["outdof_half"], dtype=int).reshape(-1)
    feedback_position_m = ((initial_positions["checkpoint"] - initial_positions["Marker100152"]) / 1000.0) + POS_END
    nodes_coord = np.asarray(rom_data["nodes_coord"], dtype=float)
    distances = np.linalg.norm(nodes_coord - feedback_position_m.reshape(3, 1), axis=0)
    node_index_1based = int(np.argmin(distances)) + 1
    outdof_feedback = np.array([3 * node_index_1based - 2, 3 * node_index_1based - 1, 3 * node_index_1based])
    return {
        "measurement_dofs_1based": np.concatenate([outdof_endeff, outdof_half, outdof_feedback]),
        "feedback_node_info": {
            "position_m": feedback_position_m,
            "node_index_1based": node_index_1based,
            "distance_m": float(distances[node_index_1based - 1]),
            "dofs_1based": outdof_feedback,
        },
    }


def save_max_displacement_sample(output_path: Path, sample: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sample, allow_pickle=True)


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def write_max_displacement_log(output_path: Path, samples: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["StaticVerification max displacement summary", ""]
    for sample in samples:
        lines.append(f"Source: {sample['source']}")
        lines.append(f"Measurement order: {', '.join(sample['measurement_order'])}")
        lines.append(f"Measurement mm: {format_vector(sample['measurement_mm'])}")
        dofs = sample.get("measurement_dofs_1based")
        if dofs is not None:
            lines.append(f"Measurement DOFs 1-based: {format_vector(dofs)}")

        for measurement_name in sample["measurement_order"]:
            item = sample["per_target"][measurement_name]
            lines.append(
                f"  {measurement_name}: "
                f"max_norm_mm={item['max_norm_mm']:.6f}, "
                f"time_s={item['time_s']:.6f}, "
                f"frame_index={item['frame_index']}, "
                f"displacement_mm={format_vector(item['displacement_mm'])}"
            )

        feedback_info = sample.get("feedback_node_info")
        if feedback_info:
            lines.append(
                "  feedback_pt node: "
                f"index_1based={feedback_info['node_index_1based']}, "
                f"distance_m={feedback_info['distance_m']:.6f}, "
                f"dofs_1based={format_vector(feedback_info['dofs_1based'])}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_file(csv_path: Path, output_dir: Path, base_y_offset: float) -> dict[str, object]:
    _, time, displacements, initial_positions = transform_to_reference_displacement(
        csv_path,
        base_y_offset=base_y_offset,
    )
    stem = csv_path.stem
    save_processed_csv(output_dir / f"{stem}_reference_displacements.csv", time, displacements)
    plot_time_curves(output_dir / f"{stem}_reference_displacements.png", stem, time, displacements)
    node_info = static_feedback_node_info(initial_positions)
    sample = extract_max_displacement_sample(
        stem,
        time,
        displacements,
        initial_positions,
        measurement_dofs_1based=node_info["measurement_dofs_1based"],
        feedback_node_info=node_info["feedback_node_info"],
    )
    save_max_displacement_sample(output_dir / f"{stem}_max_displacement_measurement.npy", sample)
    return sample


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    static_results_dir = script_dir / "results" / "Static_data"
    parser = argparse.ArgumentParser(
        description="Plot StaticVerification mideff/checkpoint centroid and Marker100152 displacement curves in Reference coordinates.",
    )
    parser.add_argument("--data-dir", type=Path, default=script_dir, help="Directory containing Static_data*.csv files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=static_results_dir,
        help="Directory for plots, processed CSV files, and static max-displacement data.",
    )
    parser.add_argument("--pattern", default="Static_data*.csv", help="Input CSV filename pattern.")
    parser.add_argument(
        "--base-y-offset",
        type=float,
        default=-12.0,
        help="Base Y offset in mm, matching initialize_motion_reference default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = sorted(args.data_dir.glob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found: {args.data_dir / args.pattern}")

    samples = []
    for csv_path in csv_files:
        sample = analyze_file(csv_path, args.output_dir, args.base_y_offset)
        samples.append(sample)
        print(f"Saved Reference displacement curves for {csv_path.name} to {args.output_dir}")
    np.save(args.output_dir / "static_verification_max_displacement_measurements.npy", samples, allow_pickle=True)
    write_max_displacement_log(args.output_dir / "static_verification_max_displacement_log.txt", samples)


if __name__ == "__main__":
    main()
