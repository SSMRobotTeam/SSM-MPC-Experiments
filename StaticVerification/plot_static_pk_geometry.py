from __future__ import annotations

import argparse
import csv
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
SOURCE_NAME = "Static_data4"
SUMMARY_SOURCE_NAMES = tuple(f"Static_data{idx}" for idx in range(2, 8))
MEASUREMENT_NAMES = ("endeff", "mideff", "checkpoint")
TABLE_NAMES = {
    "endeff": "end_marker",
    "mideff": "mideff",
    "checkpoint": "checkpoint",
}
MEASUREMENT_COLORS = {
    "endeff": "#d62728",
    "mideff": "#d62728",
    "checkpoint": "#d62728",
}
DISPLACEMENT_SCALAR_NAME = "displacement_mm"


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


def _node_index_from_dofs(dofs_1based: np.ndarray) -> int:
    dofs = np.asarray(dofs_1based, dtype=int).reshape(-1)
    node_indices = np.unique((dofs - 1) // 3)
    if node_indices.size != 1:
        raise ValueError(f"Expected one node index from measurement DOFs, got {node_indices}")
    return int(node_indices[0])


def _estimated_measurement_positions_m(
    rom_data: dict,
    nodes0_xyz: np.ndarray,
    nodes_xyz: np.ndarray,
    sample: dict,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    outdof_endeff = np.asarray(rom_data["outdof_endeff"], dtype=int).reshape(-1)
    outdof_half = np.asarray(rom_data["outdof_half"], dtype=int).reshape(-1)
    initial_positions = _measurement_initial_positions_m(sample)

    node_indices = {
        "endeff": _node_index_from_dofs(outdof_endeff),
        "mideff": _node_index_from_dofs(outdof_half),
    }

    feedback_info = sample.get("feedback_node_info")
    if feedback_info and "node_index_1based" in feedback_info:
        node_indices["checkpoint"] = int(feedback_info["node_index_1based"]) - 1
    else:
        checkpoint0 = np.asarray(initial_positions["checkpoint"], dtype=float).reshape(3)
        node_indices["checkpoint"] = int(np.argmin(np.linalg.norm(nodes0_xyz - checkpoint0, axis=1)))

    positions = {name: nodes_xyz[index] for name, index in node_indices.items()}
    return positions, node_indices


def write_measurement_summary_csv(
    output_path: Path,
    rom_data: dict,
    sample: dict,
    estimate: dict,
    null_matrix,
) -> None:
    write_measurement_summary_table(output_path, rom_data, {estimate["source"]: sample}, [estimate], null_matrix)


def _feedback_norm_mm(value) -> float | str:
    if value in ("", None):
        return ""
    try:
        return float(value) * 1000.0
    except (TypeError, ValueError):
        return ""


def _measurement_summary_rows(
    rom_data: dict,
    sample: dict,
    estimate: dict,
    null_matrix,
) -> list[dict[str, object]]:
    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    nodes_xyz = _node_xyz_from_pk(rom_data, np.asarray(estimate["pk_est"], dtype=float), null_matrix)
    estimated_positions, node_indices = _estimated_measurement_positions_m(rom_data, nodes0_xyz, nodes_xyz, sample)
    measured_positions = _measurement_positions_m(sample)

    rows = []
    for name in MEASUREMENT_NAMES:
        estimated_mm = np.asarray(estimated_positions[name], dtype=float).reshape(3) * 1000.0
        measured_mm = np.asarray(measured_positions[name], dtype=float).reshape(3) * 1000.0
        residual_mm = estimated_mm - measured_mm
        rows.append(
            {
                "source": estimate["source"],
                "node": TABLE_NAMES[name],
                "estimated_node_index_1based": node_indices[name] + 1,
                "estimated_x_mm": estimated_mm[0],
                "estimated_y_mm": estimated_mm[1],
                "estimated_z_mm": estimated_mm[2],
                "measured_x_mm": measured_mm[0],
                "measured_y_mm": measured_mm[1],
                "measured_z_mm": measured_mm[2],
                "residual_x_mm": residual_mm[0],
                "residual_y_mm": residual_mm[1],
                "residual_z_mm": residual_mm[2],
                "residual_norm_mm": float(np.linalg.norm(residual_mm)),
                "feedback_residual_norm_mm": _feedback_norm_mm(estimate.get("feedback_residual_norm", "")),
                "cable_residual_norm_mm": _feedback_norm_mm(estimate.get("cable_residual_norm", "")),
                "feedback_accepted": estimate.get("feedback_accepted", ""),
            }
        )
    return rows


def write_measurement_summary_table(
    output_path: Path,
    rom_data: dict,
    samples: dict[str, dict],
    estimates: list[dict],
    null_matrix,
    source_names: tuple[str, ...] | None = None,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "source",
        "node",
        "estimated_node_index_1based",
        "estimated_x_mm",
        "estimated_y_mm",
        "estimated_z_mm",
        "measured_x_mm",
        "measured_y_mm",
        "measured_z_mm",
        "residual_x_mm",
        "residual_y_mm",
        "residual_z_mm",
        "residual_norm_mm",
        "feedback_residual_norm_mm",
        "cable_residual_norm_mm",
        "feedback_accepted",
    ]
    estimate_by_source = {estimate["source"]: estimate for estimate in estimates}
    selected_sources = source_names or tuple(estimate_by_source)
    rows = []
    skipped = []
    for source in selected_sources:
        if source not in samples or source not in estimate_by_source:
            skipped.append(source)
            continue
        rows.extend(_measurement_summary_rows(rom_data, samples[source], estimate_by_source[source], null_matrix))

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return skipped


def _sample_from_back_verify(source: str, back_verify_data: dict) -> dict[str, object]:
    measurements = back_verify_data["steady_measurements"]
    measurement_mm = np.concatenate(
        [
            np.asarray(measurements[name]["displacement_ref_mm_mean"], dtype=float).reshape(3)
            for name in MEASUREMENT_NAMES
        ]
    )
    return {
        "source": source,
        "measurement_order": MEASUREMENT_NAMES,
        "measurement_mm": measurement_mm,
        "reference_initial_positions_mm": back_verify_data["initial_reference_positions_mm"],
        "steady_positions_ref_mm": {
            name: np.asarray(measurements[name]["position_ref_mm_mean"], dtype=float).reshape(3)
            for name in MEASUREMENT_NAMES
        },
    }


def load_back_verify_samples(data_dir: Path) -> dict[str, dict[str, object]]:
    samples = {}
    if not data_dir.exists():
        return samples
    for path in sorted(data_dir.glob("Static_data*/back_verify_measurement.npy")):
        source = path.parent.name
        samples[source] = _sample_from_back_verify(source, _load_npy_object(path))
    return samples


def _load_tet_connect(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    data = loadmat(path)
    if "tet_connect" not in data:
        raise KeyError(f"Missing variable 'tet_connect' in {path}")
    tets = np.asarray(data["tet_connect"], dtype=int)
    if tets.ndim != 2:
        raise ValueError(f"tet_connect must be a 2D array, got {tets.shape}")
    if tets.shape[1] != 4 and tets.shape[0] == 4:
        tets = tets.T
    if tets.shape[1] != 4:
        raise ValueError(f"tet_connect must have shape (M, 4), got {tets.shape}")
    if np.min(tets) == 1:
        tets = tets - 1
    return tets


def _extract_surface_from_tets(points_xyz: np.ndarray, tets: np.ndarray) -> pv.PolyData:
    faces = np.vstack(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ]
    )
    faces_sorted = np.sort(faces, axis=1)
    unique_faces, unique_indices, counts = np.unique(
        faces_sorted,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_faces = faces[unique_indices[counts == 1]]
    faces_pv = np.hstack(
        [np.full((boundary_faces.shape[0], 1), 3, dtype=np.int64), boundary_faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(np.asarray(points_xyz, dtype=float), faces_pv)


def _surface_from_nodes(nodes_xyz: np.ndarray, alpha: float, scalars: np.ndarray | None = None) -> pv.PolyData:
    cloud = pv.PolyData(nodes_xyz)
    if scalars is not None:
        cloud[DISPLACEMENT_SCALAR_NAME] = np.asarray(scalars, dtype=float).reshape(-1)
    try:
        surface = cloud.delaunay_3d(alpha=alpha).extract_surface(algorithm="dataset_surface").clean()
    except RuntimeError:
        surface = pv.PolyData()
    return surface if surface.n_points else cloud


def _geometry_surface(
    nodes_xyz: np.ndarray,
    *,
    alpha: float,
    tet_connect: np.ndarray | None,
    scalars: np.ndarray | None = None,
    smooth: bool = False,
) -> pv.PolyData:
    if tet_connect is not None:
        surface = _extract_surface_from_tets(nodes_xyz, tet_connect)
        if scalars is not None:
            surface[DISPLACEMENT_SCALAR_NAME] = np.asarray(scalars, dtype=float).reshape(-1)
    else:
        surface = _surface_from_nodes(nodes_xyz, alpha, scalars=scalars)
    if smooth:
        try:
            surface = surface.smooth(n_iter=50, relaxation_factor=0.02, feature_angle=60.0)
        except (RuntimeError, ValueError):
            pass
    return surface


def _sphere(center: np.ndarray, radius: float) -> pv.PolyData:
    return pv.Sphere(radius=radius, center=np.asarray(center, dtype=float), theta_resolution=32, phi_resolution=16)


def _plot_coordinates(xyz: np.ndarray) -> np.ndarray:
    plotted = np.asarray(xyz, dtype=float).copy()
    plotted[..., 2] *= -1.0
    return plotted


def _padded_bounds(points_xyz: np.ndarray, padding_ratio: float = 0.08) -> tuple[float, float, float, float, float, float]:
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    padding = spans * padding_ratio
    mins -= padding
    maxs += padding
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


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


def _camera_position_for_scene(
    nodes0_xyz: np.ndarray,
    nodes_xyz: np.ndarray,
    measurement_positions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera, _, viewup = _camera_position_for_deformation(nodes0_xyz, nodes_xyz)
    scene_points = np.vstack([nodes0_xyz, nodes_xyz, *measurement_positions.values()])
    focal = 0.5 * (np.min(scene_points, axis=0) + np.max(scene_points, axis=0))
    view_direction = camera - focal
    norm = np.linalg.norm(view_direction)
    if norm < 1e-9:
        view_direction = np.array([1.0, -1.0, 0.35])
        norm = np.linalg.norm(view_direction)
    view_direction = view_direction / norm
    scene_span = np.ptp(scene_points, axis=0)
    distance = 2.15 * max(float(np.linalg.norm(scene_span)), 0.25)
    return focal + distance * view_direction, focal, viewup


def _zoom_camera(plotter: pv.Plotter, factor: float) -> None:
    position, focal, viewup = plotter.camera_position
    position = np.asarray(position, dtype=float)
    focal = np.asarray(focal, dtype=float)
    new_position = focal + (position - focal) / factor
    plotter.camera_position = [tuple(new_position), tuple(focal), tuple(viewup)]


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
    tet_connect: np.ndarray | None,
    interactive_preview: bool,
) -> None:
    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    nodes_xyz = _node_xyz_from_pk(rom_data, np.asarray(estimate["pk_est"], dtype=float), null_matrix)
    measurement_positions = _measurement_positions_m(sample)
    nodes0_plot = _plot_coordinates(nodes0_xyz)
    nodes_plot = _plot_coordinates(nodes_xyz)
    measurement_positions_plot = {name: _plot_coordinates(position) for name, position in measurement_positions.items()}

    pv.OFF_SCREEN = not interactive_preview
    plotter = pv.Plotter(off_screen=not interactive_preview, window_size=(1800, 2400))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_trackball_style()

    reference_surface = _geometry_surface(
        nodes0_plot,
        alpha=delaunay_alpha,
        tet_connect=tet_connect,
        smooth=False,
    )
    plotter.add_mesh(
        reference_surface,
        color="#b8b8b8",
        opacity=reference_alpha,
        smooth_shading=True,
        show_edges=False,
    )

    surface = _geometry_surface(
        nodes_plot,
        alpha=delaunay_alpha,
        tet_connect=tet_connect,
        smooth=True,
    )
    plotter.add_mesh(
        surface,
        color="#7aa6c2",
        opacity=surface_alpha,
        smooth_shading=True,
        show_edges=True,
        edge_color="#004f8f",
        line_width=0.3,
        specular=0.25,
        specular_power=18,
    )
    marker_radius = 0.0045
    for name in sample.get("measurement_order", MEASUREMENT_NAMES):
        color = MEASUREMENT_COLORS[name]
        measured = measurement_positions_plot[name]

        plotter.add_mesh(_sphere(measured, marker_radius), color=color, opacity=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter.camera_position = _camera_position_for_scene(nodes0_plot, nodes_plot, measurement_positions_plot)
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(1.18)
    if interactive_preview:
        def save_current_view() -> None:
            plotter.screenshot(str(output_path))
            print(f"Saved {output_path}")
            plotter.close()

        print(
            f"Adjust the view freely for {output_path.name}, then press 's' in the PyVista window "
            "to save the current camera state and close."
        )
        plotter.add_key_event("s", save_current_view)
        plotter.show(auto_close=False, interactive=True)
        if not output_path.exists():
            print(f"No image saved. Re-run and press 's' in the PyVista window, or use --off-screen: {output_path}")
    else:
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
    parser.add_argument(
        "--back-verify-dir",
        type=Path,
        default=static_results_dir,
        help="Directory containing Static_data*/back_verify_measurement.npy files used for measured marker positions.",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results" / "geometry_compare")
    parser.add_argument(
        "--source",
        default=SOURCE_NAME,
        help="Source to plot. Defaults to SOURCE_NAME at the top of this file.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Also plot pk estimates whose feedback_accepted flag is False.",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Save images directly without opening an interactive PyVista preview window.",
    )
    parser.add_argument("--surface-alpha", type=float, default=0.4, help="Transparency of reconstructed arm geometry.")
    parser.add_argument(
        "--reference-alpha",
        type=float,
        default=0.16,
        help="Transparency of undeformed nodes_coord reference surface.",
    )
    parser.add_argument(
        "--null",
        type=Path,
        default=None,
        help="Optional legacy Null matrix. By default constrained nodes_coord is used directly.",
    )
    parser.add_argument(
        "--tet-connect",
        type=Path,
        default=script_dir / "tet_connect.mat",
        help="Optional FEM tetrahedral connectivity file. If present, boundary faces are extracted from tet_connect.",
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
    tet_connect = _load_tet_connect(args.tet_connect)
    if tet_connect is None:
        print(f"No tet_connect file found at {args.tet_connect}; falling back to Delaunay surface reconstruction.")
    samples = {sample["source"]: sample for sample in _as_list(_load_npy_object(args.measurements))}
    samples.update(load_back_verify_samples(args.back_verify_dir))
    estimates = _as_list(_load_npy_object(args.pk_estimates))
    if not args.include_rejected:
        estimates = [estimate for estimate in estimates if bool(estimate.get("feedback_accepted", True))]
    combined_summary_path = args.output_dir / "Static_data2_to_Static_data7_estimated_measurement_summary.csv"
    skipped_sources = write_measurement_summary_table(
        combined_summary_path,
        rom_data,
        samples,
        estimates,
        null_matrix,
        SUMMARY_SOURCE_NAMES,
    )
    print(f"Saved Static_data2~7 measurement summary to {combined_summary_path}")
    if skipped_sources:
        print(f"Skipped missing summary sources: {', '.join(skipped_sources)}")
    source = args.source
    matching = [estimate for estimate in estimates if estimate["source"] == source]
    if not matching:
        raise ValueError(f"No accepted pk estimate found for source: {source}")

    if source not in samples:
        raise KeyError(f"No measurement sample found for source: {source}")
    output_path = args.output_dir / f"{source}_ssm_geometry_measurement_compare.png"
    summary_path = args.output_dir / f"{source}_estimated_measurement_summary.csv"
    write_measurement_summary_csv(
        summary_path,
        rom_data,
        samples[source],
        matching[0],
        null_matrix,
    )
    print(f"Saved measurement summary to {summary_path}")
    plot_sample_geometry(
        output_path,
        rom_data,
        samples[source],
        matching[0],
        null_matrix=null_matrix,
        surface_alpha=args.surface_alpha,
        delaunay_alpha=args.delaunay_alpha,
        reference_alpha=args.reference_alpha,
        tet_connect=tet_connect,
        interactive_preview=not args.off_screen,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
