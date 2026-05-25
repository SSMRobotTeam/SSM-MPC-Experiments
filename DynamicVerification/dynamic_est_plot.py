from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 18,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Main_demo_SSM import POS_END, SSM_FACTOR
from dynamics.SSM.matlab_data import ROM_DIR, clone_struct_array, get_field, load_rom_data, loadmat
from dynamics.SSM.ssm_dynamics import reduced_to_full


MARKER_NAMES = ("endeff",)
AXES = ("x", "y", "z")
MARKER_COLORS = {
    "endeff": "#d62728",
}
MEASURED_COLOR = "#0072B2"
SSM_COLOR = "#D55E00"
PCS_COLOR = "#009E73"
CABLE_COLORS = ("#0072B2", "#D55E00", "#009E73")
AXIS_COLORS = {
    "x": "#d62728",
    "y": "#2ca02c",
    "z": "#1f77b4",
}


def _load_npy_object(path: Path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    return data.tolist()


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


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


def _require_columns(data: dict[str, np.ndarray], columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise KeyError(f"{label} is missing required columns: {', '.join(missing)}")


def _validate_lengths(raw_data: dict[str, np.ndarray], est_data: dict[str, np.ndarray]) -> None:
    raw_len = len(raw_data["step_index"])
    est_len = len(est_data["row_index"])
    if raw_len != est_len:
        raise ValueError(f"Row count mismatch: raw data has {raw_len}, estimate data has {est_len}.")


def _validate_optional_lengths(raw_data: dict[str, np.ndarray], data: dict[str, np.ndarray], label: str) -> None:
    raw_len = len(raw_data["step_index"])
    data_len = len(data["row_index"])
    if raw_len != data_len:
        raise ValueError(f"Row count mismatch: raw data has {raw_len}, {label} has {data_len}.")


def _time_vector(raw_data: dict[str, np.ndarray], est_data: dict[str, np.ndarray], preferred: str) -> np.ndarray:
    if preferred in est_data:
        return np.asarray(est_data[preferred], dtype=float)
    if preferred in raw_data:
        return np.asarray(raw_data[preferred], dtype=float)
    raise KeyError(f"Missing time column: {preferred}")


def _measured_displacement(raw_data: dict[str, np.ndarray], marker_name: str) -> np.ndarray:
    columns = [f"{marker_name}_disp_{axis}_ref_mm" for axis in AXES]
    _require_columns(raw_data, columns, "Dyndata CSV")
    displacement = np.column_stack([np.asarray(raw_data[column], dtype=float) for column in columns])
    if marker_name == "endeff":
        displacement[:, 0] *= -1.0
    return displacement


def _pcs_displacement(pcs_data: dict[str, np.ndarray], marker_name: str) -> np.ndarray:
    columns = [f"pcs_{marker_name}_displacement_{axis}_mm" for axis in AXES]
    _require_columns(pcs_data, columns, "PCS estimate CSV")
    return np.column_stack([np.asarray(pcs_data[column], dtype=float) for column in columns])


def _pcs_position(pcs_data: dict[str, np.ndarray], marker_name: str) -> np.ndarray:
    columns = [f"pcs_{marker_name}_position_{axis}_mm" for axis in AXES]
    _require_columns(pcs_data, columns, "PCS estimate CSV")
    return np.column_stack([np.asarray(pcs_data[column], dtype=float) for column in columns])


def _cable_matrix(data: dict[str, np.ndarray], prefix: str, label: str) -> np.ndarray:
    columns = [f"{prefix}_{idx + 1}" for idx in range(6)]
    _require_columns(data, columns, label)
    return np.column_stack([np.asarray(data[column], dtype=float) for column in columns])


def _pk_matrix(est_data: dict[str, np.ndarray]) -> np.ndarray:
    pk_columns = sorted(
        (name for name in est_data if name.startswith("pk_")),
        key=lambda name: int(name.split("_", maxsplit=1)[1]),
    )
    if not pk_columns:
        raise KeyError("estimate CSV is missing pk_* columns.")
    return np.column_stack([np.asarray(est_data[column], dtype=float) for column in pk_columns])


def _scaled_w0(rom_data: dict):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR
    return w0


def _load_null_matrix(path: Path):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Null constraint matrix not found: {path}")
    data = loadmat(path)
    if "Null" not in data:
        raise KeyError(f"Missing variable 'Null' in {path}")
    return data["Null"]


def _full_node_displacement(reduced_displacement: np.ndarray, node_count: int, null_matrix) -> np.ndarray:
    reduced_displacement = np.asarray(reduced_displacement, dtype=float).reshape(-1)
    full_size = 3 * node_count
    if reduced_displacement.size == full_size:
        return reduced_displacement
    if null_matrix is None:
        raise ValueError(
            f"SSM displacement has {reduced_displacement.size} DOFs, but nodes_coord needs {full_size}. "
            "Provide --null to expand constrained DOFs."
        )

    null_shape = getattr(null_matrix, "shape", None)
    if null_shape != (full_size, reduced_displacement.size):
        raise ValueError(
            f"Null matrix shape {null_shape} does not match full/reduced DOFs "
            f"({full_size}, {reduced_displacement.size})."
        )
    return np.asarray(null_matrix @ reduced_displacement, dtype=float).reshape(-1)


def _node_xyz_from_pk(rom_data: dict, pk: np.ndarray, null_matrix, w0_scaled) -> np.ndarray:
    nodes0 = np.asarray(rom_data["nodes_coord"], dtype=float).T
    ssm_state = reduced_to_full(pk, w0_scaled, None, 0).reshape(-1)
    reduced_size = null_matrix.shape[1] if null_matrix is not None else 3 * nodes0.shape[0]
    if ssm_state.size < reduced_size:
        raise ValueError(f"SSM state has {ssm_state.size} values, expected at least {reduced_size} displacement DOFs.")
    displacement_xyz = _full_node_displacement(ssm_state[:reduced_size], nodes0.shape[0], null_matrix).reshape(-1, 3)
    return nodes0 + displacement_xyz


def _node_index_from_dofs(dofs_1based: np.ndarray) -> int:
    dofs = np.asarray(dofs_1based, dtype=int).reshape(-1)
    node_indices = np.unique((dofs - 1) // 3)
    if node_indices.size != 1:
        raise ValueError(f"Expected one node index from measurement DOFs, got {node_indices}")
    return int(node_indices[0])


def _reference_positions_from_file(path: Path) -> dict[str, np.ndarray]:
    loaded = _as_list(_load_npy_object(path))
    for item in loaded:
        if isinstance(item, dict) and "reference_initial_positions_mm" in item:
            positions = item["reference_initial_positions_mm"]
            return {"endeff": np.asarray(positions["endeff"], dtype=float).reshape(3)}
        if isinstance(item, dict) and "initial_reference_positions_mm" in item:
            positions = item["initial_reference_positions_mm"]
            return {"endeff": np.asarray(positions["endeff"], dtype=float).reshape(3)}
    raise KeyError(f"{path} does not contain reference_initial_positions_mm or initial_reference_positions_mm")


def _initial_positions_m(reference_initial_positions_mm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    endeff0 = np.asarray(POS_END, dtype=float).reshape(3)
    return {"endeff": endeff0}


def _measurement_node_indices(
    rom_data: dict,
    reference_initial_positions_mm: dict[str, np.ndarray],
) -> dict[str, int]:
    return {
        "endeff": _node_index_from_dofs(np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)),
    }


def _ssm_positions_from_pk(
    est_data: dict[str, np.ndarray],
    rom_data: dict,
    null_matrix,
    w0_scaled,
    node_indices: dict[str, int],
) -> dict[str, np.ndarray]:
    pk_values = _pk_matrix(est_data)
    positions = {name: [] for name in MARKER_NAMES}
    for pk in pk_values:
        nodes_xyz = _node_xyz_from_pk(rom_data, pk, null_matrix, w0_scaled)
        for name in MARKER_NAMES:
            positions[name].append(nodes_xyz[node_indices[name]] * 1000.0)
    return {name: np.asarray(values, dtype=float) for name, values in positions.items()}


def _measured_positions_from_displacement(
    raw_data: dict[str, np.ndarray],
    initial_positions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: initial_positions[name].reshape(1, 3) * 1000.0 + _measured_displacement(raw_data, name)
        for name in MARKER_NAMES
    }


def _displacements_from_positions(
    positions_mm: dict[str, np.ndarray],
    initial_positions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: positions_mm[name] - initial_positions[name].reshape(1, 3) * 1000.0
        for name in MARKER_NAMES
    }


def _set_equal_3d_axes(ax, points: list[np.ndarray]) -> None:
    stacked = np.vstack([point.reshape(-1, 3) for point in points if point.size])
    center = 0.5 * (np.min(stacked, axis=0) + np.max(stacked, axis=0))
    radius = 0.5 * float(np.max(np.ptp(stacked, axis=0)))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_3d_displacement_overlay(
    raw_data: dict[str, np.ndarray],
    ssm_displacements: dict[str, np.ndarray],
    pcs_displacements: dict[str, np.ndarray] | None,
    output_path: Path,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plotted_points = []

    ax.scatter(0.0, 0.0, 0.0, color="black", marker="x", s=70, linewidths=2.0, label="origin")
    for marker_name in MARKER_NAMES:
        measured = _measured_displacement(raw_data, marker_name)
        ssm = ssm_displacements[marker_name]
        plotted_points.extend([measured, ssm])
        pcs = pcs_displacements[marker_name] if pcs_displacements is not None else None
        if pcs is not None:
            plotted_points.append(pcs)

        ax.plot(
            measured[:, 0],
            measured[:, 1],
            measured[:, 2],
            color=MEASURED_COLOR,
            linewidth=2.0,
            label=f"{marker_name} measured",
        )
        ax.plot(
            ssm[:, 0],
            ssm[:, 1],
            ssm[:, 2],
            color=SSM_COLOR,
            linestyle="--",
            linewidth=2.0,
            label=f"{marker_name} SSM",
        )
        if pcs is not None:
            ax.plot(
                pcs[:, 0],
                pcs[:, 1],
                pcs[:, 2],
                color=PCS_COLOR,
                linestyle="-.",
                linewidth=2.0,
                label=f"{marker_name} PCS",
            )

    _set_equal_3d_axes(ax, plotted_points + [np.zeros((1, 3))])
    ax.set_title("Measured vs SSM/PCS 3D Trajectories")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=300)
    return fig


def plot_3d_position_overlay(
    measured_positions: dict[str, np.ndarray],
    ssm_positions: dict[str, np.ndarray],
    pcs_positions: dict[str, np.ndarray] | None,
    output_path: Path,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plotted_points = []

    for marker_name in MARKER_NAMES:
        measured = measured_positions[marker_name]
        ssm = ssm_positions[marker_name]
        plotted_points.extend([measured, ssm])
        pcs = pcs_positions[marker_name] if pcs_positions is not None else None
        if pcs is not None:
            plotted_points.append(pcs)

        ax.plot(
            measured[:, 0],
            measured[:, 1],
            measured[:, 2],
            color=MEASURED_COLOR,
            linewidth=2.0,
            label=f"{marker_name} measured",
        )
        ax.plot(
            ssm[:, 0],
            ssm[:, 1],
            ssm[:, 2],
            color=SSM_COLOR,
            linestyle="--",
            linewidth=2.0,
            label=f"{marker_name} SSM",
        )
        if pcs is not None:
            ax.plot(
                pcs[:, 0],
                pcs[:, 1],
                pcs[:, 2],
                color=PCS_COLOR,
                linestyle="-.",
                linewidth=2.0,
                label=f"{marker_name} PCS",
            )

    _set_equal_3d_axes(ax, plotted_points)
    ax.set_title("Measured vs SSM/PCS 3D Position Trajectories")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_zlabel("Z position (mm)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=300)
    return fig


def plot_marker_time_overlay(
    marker_name: str,
    time_s: np.ndarray,
    raw_data: dict[str, np.ndarray],
    ssm_displacements: dict[str, np.ndarray],
    pcs_displacements: dict[str, np.ndarray] | None,
    output_path: Path,
) -> plt.Figure:
    measured = _measured_displacement(raw_data, marker_name)
    ssm = ssm_displacements[marker_name]
    pcs = pcs_displacements[marker_name] if pcs_displacements is not None else None

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for idx, axis in enumerate(AXES):
        axes[idx].plot(time_s, measured[:, idx], color=MEASURED_COLOR, linewidth=1.8, label=f"{marker_name} measured")
        axes[idx].plot(time_s, ssm[:, idx], color=SSM_COLOR, linestyle="--", linewidth=1.6, label=f"{marker_name} SSM")
        if pcs is not None:
            axes[idx].plot(time_s, pcs[:, idx], color=PCS_COLOR, linestyle="-.", linewidth=1.6, label=f"{marker_name} PCS")
        axes[idx].set_ylabel(f"{axis.upper()} (mm)")
        axes[idx].grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axes[idx].legend(loc="best")

    axes[0].set_title(f"{marker_name} Measured vs SSM/PCS")
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(output_path, dpi=300)
    return fig


def plot_cable_control_overlay(
    time_s: np.ndarray,
    measured_cables: np.ndarray,
    ssm_cables: np.ndarray,
    pcs_cables: np.ndarray | None,
    cable_indices: tuple[int, int, int],
    title: str,
    output_path: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for axis_idx, cable_idx in enumerate(cable_indices):
        axes[axis_idx].plot(
            time_s,
            measured_cables[:, cable_idx],
            color=MEASURED_COLOR,
            linewidth=1.8,
            label="measured",
        )
        axes[axis_idx].plot(
            time_s,
            ssm_cables[:, cable_idx],
            color=SSM_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="SSM",
        )
        if pcs_cables is not None:
            axes[axis_idx].plot(
                time_s,
                pcs_cables[:, cable_idx],
                color=PCS_COLOR,
                linestyle="-.",
                linewidth=1.6,
                label="PCS",
            )
        axes[axis_idx].set_ylabel(f"Cable {cable_idx + 1} (mm)")
        axes[axis_idx].grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axes[axis_idx].legend(loc="best")

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(output_path, dpi=300)
    return fig


def _total_ise(time_s: np.ndarray, errors: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=float)
    if time_s.size <= 1:
        return 0.0
    per_axis = np.trapezoid(errors**2, x=time_s, axis=0)
    return float(np.sum(per_axis))


def plot_total_ise_bar(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = [SSM_COLOR if label == "SSM" else PCS_COLOR for label in labels]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4g}",
            ha="center",
            va="bottom",
        )
    fig.savefig(output_path, dpi=300)
    return fig


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_reference = (
        REPO_ROOT
        / "StaticVerification"
        / "results"
        / "Static_data"
        / "static_verification_max_displacement_measurements.npy"
    )
    parser = argparse.ArgumentParser(
        description="Plot Dyndata01 measured node trajectories against SSM trajectories estimated from pk.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "DynData" / "Dyndata01.csv",
        help="Input Dyndata CSV.",
    )
    parser.add_argument(
        "--estimate",
        type=Path,
        default=script_dir / "results" / "dynamic_pk_estimates.csv",
        help="CSV generated by estimate_dynamic_pk.py.",
    )
    parser.add_argument(
        "--pcs-estimate",
        type=Path,
        default=script_dir / "results" / "pcs_estimates.csv",
        help="CSV generated by estimate_pcs.py. If missing, PCS curves are skipped.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=default_reference,
        help="Npy containing reference_initial_positions_mm, used to locate checkpoint exactly like plot_static_pk_geometry.py.",
    )
    parser.add_argument(
        "--null",
        type=Path,
        default=None,
        help="Optional legacy Null matrix. By default constrained nodes_coord is used directly.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "dynamic_est_plot",
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--time-column",
        default="actual_time_s",
        choices=("actual_time_s", "planned_time_s"),
        help="Time column used for time-series plots.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_data = _load_csv(args.data)
    est_data = _load_csv(args.estimate)
    _validate_lengths(raw_data, est_data)
    pcs_data = None
    if args.pcs_estimate.exists():
        pcs_data = _load_csv(args.pcs_estimate)
        _validate_optional_lengths(raw_data, pcs_data, "PCS estimate CSV")
    else:
        print(f"PCS estimate file not found, skipping PCS curves: {args.pcs_estimate}")
    time_s = _time_vector(raw_data, est_data, args.time_column)
    reference_initial_positions_mm = _reference_positions_from_file(args.reference)
    initial_positions = _initial_positions_m(reference_initial_positions_mm)
    rom_data = load_rom_data(ROM_DIR)
    null_matrix = _load_null_matrix(args.null)
    w0_scaled = _scaled_w0(rom_data)
    print(f"Using SSM_FACTOR={SSM_FACTOR} for W_0[1].coeffs, matching Main_demo_SSM.build_ssm_maps().")
    node_indices = _measurement_node_indices(rom_data, reference_initial_positions_mm)
    print(
        "Measurement node indices, 1-based: "
        + ", ".join(f"{name}={index + 1}" for name, index in node_indices.items())
    )
    ssm_positions = _ssm_positions_from_pk(est_data, rom_data, null_matrix, w0_scaled, node_indices)
    ssm_displacements = _displacements_from_positions(ssm_positions, initial_positions)
    measured_positions = _measured_positions_from_displacement(raw_data, initial_positions)
    pcs_displacements = None
    pcs_positions = None
    if pcs_data is not None:
        pcs_displacements = {name: _pcs_displacement(pcs_data, name) for name in MARKER_NAMES}
        pcs_positions = {name: _pcs_position(pcs_data, name) for name in MARKER_NAMES}
    measured_cables = _cable_matrix(est_data, "cable_target_mm", "SSM estimate CSV")
    ssm_cables = _cable_matrix(est_data, "cable_delta_mm", "SSM estimate CSV")
    pcs_cables = _cable_matrix(pcs_data, "cable_delta_mm", "PCS estimate CSV") if pcs_data is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    displacement_path = args.output_dir / "Dyndata01_vs_SSM_PCS_endeff_3d.png"
    figures.append(plot_3d_displacement_overlay(raw_data, ssm_displacements, pcs_displacements, displacement_path))
    print(f"Saved {displacement_path}")

    position_path = args.output_dir / "Dyndata01_vs_SSM_PCS_endeff_3d_position.png"
    figures.append(plot_3d_position_overlay(measured_positions, ssm_positions, pcs_positions, position_path))
    print(f"Saved {position_path}")

    for marker_name in MARKER_NAMES:
        output_path = args.output_dir / f"Dyndata01_vs_SSM_PCS_{marker_name}_time.png"
        figures.append(plot_marker_time_overlay(marker_name, time_s, raw_data, ssm_displacements, pcs_displacements, output_path))
        print(f"Saved {output_path}")

    long_cable_path = args.output_dir / "Dyndata01_vs_SSM_PCS_long_cables.png"
    figures.append(
        plot_cable_control_overlay(
            time_s,
            measured_cables,
            ssm_cables,
            pcs_cables,
            (0, 1, 2),
            "Long Cable Control Comparison",
            long_cable_path,
        )
    )
    print(f"Saved {long_cable_path}")

    short_cable_path = args.output_dir / "Dyndata01_vs_SSM_PCS_short_cables.png"
    figures.append(
        plot_cable_control_overlay(
            time_s,
            measured_cables,
            ssm_cables,
            pcs_cables,
            (3, 4, 5),
            "Short Cable Control Comparison",
            short_cable_path,
        )
    )
    print(f"Saved {short_cable_path}")

    measured_endeff = _measured_displacement(raw_data, "endeff")
    ssm_cable_total_ise = _total_ise(time_s, ssm_cables - measured_cables)
    ssm_endeff_total_ise = _total_ise(time_s, ssm_displacements["endeff"] - measured_endeff)
    ise_labels = ["SSM"]
    cable_ise_values = [ssm_cable_total_ise]
    endeff_ise_values = [ssm_endeff_total_ise]
    if pcs_cables is not None and pcs_displacements is not None:
        ise_labels.append("PCS")
        cable_ise_values.append(_total_ise(time_s, pcs_cables - measured_cables))
        endeff_ise_values.append(_total_ise(time_s, pcs_displacements["endeff"] - measured_endeff))

    cable_ise_path = args.output_dir / "Dyndata01_SSM_PCS_cable_total_ISE.png"
    figures.append(
        plot_total_ise_bar(
            ise_labels,
            cable_ise_values,
            title="Cable Total ISE",
            ylabel=r"Total ISE ($\mathrm{mm}^2\cdot\mathrm{s}$)",
            output_path=cable_ise_path,
        )
    )
    print(f"Saved {cable_ise_path}")

    endeff_ise_path = args.output_dir / "Dyndata01_SSM_PCS_endeff_total_ISE.png"
    figures.append(
        plot_total_ise_bar(
            ise_labels,
            endeff_ise_values,
            title="Endeff Total ISE",
            ylabel=r"Total ISE ($\mathrm{mm}^2\cdot\mathrm{s}$)",
            output_path=endeff_ise_path,
        )
    )
    print(f"Saved {endeff_ise_path}")

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
