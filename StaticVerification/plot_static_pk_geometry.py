from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamics.SSM.matlab_data import ROM_DIR, clone_struct_array, get_field, load_rom_data, loadmat
from dynamics.SSM.ssm_dynamics import reduced_to_full


SSM_FACTOR = 1.475
POS_END_M = np.array([0.0, 0.0, 0.4400])
MEASUREMENT_NAMES = ("endeff", "mideff", "checkpoint")
MEASUREMENT_COLORS = {
    "endeff": "#d62728",
    "mideff": "#1f77b4",
    "checkpoint": "#2ca02c",
}


def _load_npy_object(path: Path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    return data.tolist()


def _as_list(value) -> list[dict]:
    return value if isinstance(value, list) else [value]


def _scaled_w0(rom_data: dict):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR
    return w0


def _load_null_matrix(path: Path):
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


def _node_xyz_from_pk(rom_data: dict, pk: np.ndarray, null_matrix) -> np.ndarray:
    nodes0 = np.asarray(rom_data["nodes_coord"], dtype=float).T
    ssm_state = reduced_to_full(pk, _scaled_w0(rom_data), None, 0).reshape(-1)
    reduced_size = null_matrix.shape[1] if null_matrix is not None else 3 * nodes0.shape[0]
    if ssm_state.size < reduced_size:
        raise ValueError(f"SSM state has {ssm_state.size} values, expected at least {reduced_size} displacement DOFs.")
    displacement_xyz = _full_node_displacement(ssm_state[:reduced_size], nodes0.shape[0], null_matrix).reshape(-1, 3)
    return nodes0 + displacement_xyz


def _measurement_initial_positions_m(sample: dict) -> dict[str, np.ndarray]:
    initial_mm = sample["reference_initial_positions_mm"]
    endeff0 = POS_END_M
    return {
        "endeff": endeff0,
        "mideff": ((np.asarray(initial_mm["mideff"], dtype=float) - np.asarray(initial_mm["endeff"], dtype=float)) / 1000.0)
        + endeff0,
        "checkpoint": (
            (np.asarray(initial_mm["checkpoint"], dtype=float) - np.asarray(initial_mm["endeff"], dtype=float)) / 1000.0
        )
        + endeff0,
    }


def _measurement_positions_m(sample: dict) -> dict[str, np.ndarray]:
    initial = _measurement_initial_positions_m(sample)
    measurement_mm = np.asarray(sample["measurement_mm"], dtype=float).reshape(3, 3)
    return {
        name: initial[name] + measurement_mm[idx] / 1000.0
        for idx, name in enumerate(sample.get("measurement_order", MEASUREMENT_NAMES))
    }


def _surface_from_nodes(nodes_xyz: np.ndarray, alpha: float) -> pv.PolyData:
    cloud = pv.PolyData(nodes_xyz)
    try:
        surface = cloud.delaunay_3d(alpha=alpha).extract_surface(algorithm="dataset_surface").clean()
    except RuntimeError:
        surface = pv.PolyData()
    return surface if surface.n_points else cloud


def _sphere(center: np.ndarray, radius: float) -> pv.PolyData:
    return pv.Sphere(radius=radius, center=np.asarray(center, dtype=float), theta_resolution=32, phi_resolution=16)


def _plot_coordinates(xyz: np.ndarray) -> np.ndarray:
    plotted = np.asarray(xyz, dtype=float).copy()
    plotted[..., 2] *= -1.0
    return plotted


def _centerline(nodes0_xyz: np.ndarray, nodes_xyz: np.ndarray, section_count: int = 36) -> tuple[np.ndarray, np.ndarray]:
    axial = nodes0_xyz[:, 2]
    z_edges = np.linspace(float(np.min(axial)), float(np.max(axial)), section_count + 1)
    center0 = []
    center = []
    for lo, hi in zip(z_edges[:-1], z_edges[1:]):
        mask = (axial >= lo) & (axial <= hi)
        if np.any(mask):
            center0.append(np.mean(nodes0_xyz[mask], axis=0))
            center.append(np.mean(nodes_xyz[mask], axis=0))
    return np.asarray(center0), np.asarray(center)


def _camera_position_for_deformation(nodes0_xyz: np.ndarray, nodes_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center0, center = _centerline(nodes0_xyz, nodes_xyz)
    bounds = np.vstack([nodes0_xyz, nodes_xyz])
    focal = np.mean(bounds, axis=0)
    span = np.ptp(bounds, axis=0)
    distance = 2.8 * max(float(np.linalg.norm(span)), 0.25)

    if center.shape[0] < 2:
        return focal + np.array([distance, -distance, 0.6 * distance]), focal, np.array([0.0, 0.0, 1.0])

    axial = center0[-1] - center0[0]
    axial_norm = np.linalg.norm(axial)
    axial = axial / axial_norm if axial_norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    deformation = center - center0
    lateral = deformation - (deformation @ axial).reshape(-1, 1) * axial.reshape(1, 3)
    max_idx = int(np.argmax(np.linalg.norm(lateral, axis=1)))
    bend = lateral[max_idx]
    bend_norm = np.linalg.norm(bend)
    if bend_norm < 1e-7:
        bend = np.array([1.0, 0.0, 0.0])
    else:
        bend = bend / bend_norm

    view_normal = np.cross(axial, bend)
    normal_norm = np.linalg.norm(view_normal)
    if normal_norm < 1e-7:
        view_normal = np.array([1.0, -1.0, 0.35])
    else:
        view_normal = view_normal / normal_norm

    camera = focal + distance * view_normal + 0.2 * distance * axial
    return camera, focal, axial


def plot_sample_geometry(
    output_path: Path,
    rom_data: dict,
    sample: dict,
    estimate: dict,
    *,
    null_matrix,
    surface_alpha: float,
    delaunay_alpha: float,
    reference_alpha: float,
) -> None:
    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    nodes_xyz = _node_xyz_from_pk(rom_data, np.asarray(estimate["pk_est"], dtype=float), null_matrix)
    measurement_positions = _measurement_positions_m(sample)
    nodes0_plot = _plot_coordinates(nodes0_xyz)
    nodes_plot = _plot_coordinates(nodes_xyz)
    measurement_positions_plot = {name: _plot_coordinates(position) for name, position in measurement_positions.items()}

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 2400))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")

    reference_surface = _surface_from_nodes(nodes0_plot, alpha=delaunay_alpha)
    plotter.add_mesh(
        reference_surface,
        color="#b8b8b8",
        opacity=reference_alpha,
        smooth_shading=True,
        show_edges=True,
        edge_color="#8f8f8f",
        line_width=0.2,
    )

    surface = _surface_from_nodes(nodes_plot, alpha=delaunay_alpha)
    plotter.add_mesh(
        surface,
        color="#7aa6c2",
        opacity=surface_alpha,
        smooth_shading=True,
        show_edges=True,
        edge_color="#54798f",
        line_width=0.25,
    )
    marker_radius = 0.0045
    for name in sample.get("measurement_order", MEASUREMENT_NAMES):
        color = MEASUREMENT_COLORS[name]
        measured = measurement_positions_plot[name]

        plotter.add_mesh(_sphere(measured, marker_radius), color=color, opacity=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter.show_bounds(
        grid="front",
        location="outer",
        color="#555555",
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="-Z (m)",
        n_xlabels=6,
        n_ylabels=6,
        n_zlabels=7,
        font_size=14,
        bold=False,
        use_2d=False,
    )
    plotter.camera_position = _camera_position_for_deformation(nodes0_plot, nodes_plot)
    plotter.camera.zoom(0.9)
    plotter.screenshot(str(output_path))
    plotter.close()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    static_results_dir = script_dir / "results" / "Static_data"
    parser = argparse.ArgumentParser(
        description="Plot transparent 3D SSM geometry and measured nodes for StaticVerification pk estimates.",
    )
    parser.add_argument(
        "--pk-estimates",
        type=Path,
        default=script_dir / "results" / "static_verification_pk_estimates_v2.npy",
        help="Input npy generated by estimate_static_pk.py.",
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=static_results_dir / "static_verification_max_displacement_measurements.npy",
        help="Input npy generated by plot_static_verification.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results" / "geometry_compare")
    parser.add_argument("--source", help="Only plot one source, for example Static_data3.")
    parser.add_argument("--surface-alpha", type=float, default=0.32, help="Transparency of reconstructed arm geometry.")
    parser.add_argument(
        "--reference-alpha",
        type=float,
        default=0.16,
        help="Transparency of undeformed nodes_coord reference surface.",
    )
    parser.add_argument(
        "--null",
        type=Path,
        default=script_dir / "Null.mat",
        help="Constraint null-space matrix that maps reduced SSM displacement DOFs to all nodes_coord DOFs.",
    )
    parser.add_argument(
        "--delaunay-alpha",
        type=float,
        default=0.035,
        help="PyVista 3D Delaunay alpha radius in meters for nodes_coord surface reconstruction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rom_data = load_rom_data(ROM_DIR)
    null_matrix = _load_null_matrix(args.null)
    samples = {sample["source"]: sample for sample in _as_list(_load_npy_object(args.measurements))}
    estimates = _as_list(_load_npy_object(args.pk_estimates))
    if args.source:
        estimates = [estimate for estimate in estimates if estimate["source"] == args.source]
        if not estimates:
            raise ValueError(f"No pk estimate found for source: {args.source}")

    for estimate in estimates:
        source = estimate["source"]
        if source not in samples:
            raise KeyError(f"No measurement sample found for source: {source}")
        output_path = args.output_dir / f"{source}_ssm_geometry_measurement_compare.png"
        plot_sample_geometry(
            output_path,
            rom_data,
            samples[source],
            estimate,
            null_matrix=null_matrix,
            surface_alpha=args.surface_alpha,
            delaunay_alpha=args.delaunay_alpha,
            reference_alpha=args.reference_alpha,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
