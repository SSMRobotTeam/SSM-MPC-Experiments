from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sciIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Main_demo_ssm_avoid_complex import (  # noqa: E402
    MODEL_INFO_DIR,
    OUTPUT_SUFFIX,
    SSM_FACTOR,
    TRAJECTORY_NAME,
    resolve_trajectory_dir,
)
from dynamics.SSM.matlab_data import clone_struct_array, get_field, load_rom_data  # noqa: E402
from dynamics.SSM.ssm_dynamics import reduced_to_full  # noqa: E402


# Edit this list for quick debugging. These are zero-based row indices in pk_save_ssm_avoid_complex.npy.
USER_DEFINED_PK_INDICES = [360]

DEFAULT_OBSTACLE_RADIUS_M = 0.015
DEFAULT_POS_END_M = np.array([0.0, 0.0, 0.4400])
GEOMETRY_FIGURE_DIR_NAME = "geometry_figures"
DISPLACEMENT_SCALAR_NAME = "displacement_mm"
GEOMETRY_INTERACTIVE_VIEW = True
GEOMETRY_MIRROR_X_AXIS = False
GEOMETRY_MIRROR_Y_AXIS = True
GEOMETRY_REVERSE_Z_AXIS = True
GEOMETRY_WINDOW_SIZE = (1100, 900)
GEOMETRY_PARALLEL_SCALE_FACTOR = 0.46
GEOMETRY_CROP_MARGIN_PX = 12
OBSTACLE_DISPLAY_OFFSET_M = 0.020


def _result_path(result_dir: Path, stem: str) -> Path:
    return result_dir / f"{stem}{OUTPUT_SUFFIX}.npy"


def load_result_data(result_dir: Path) -> dict[str, np.ndarray]:
    result_dir = Path(result_dir)
    pk_path = _result_path(result_dir, "pk_save_ssm_avoid")
    if not pk_path.exists():
        raise FileNotFoundError(f"pk_save file not found: {pk_path}")

    t_path = _result_path(result_dir, "t_ssm_avoid")
    data = {"pk_save": np.load(pk_path)}
    data["t"] = np.load(t_path) if t_path.exists() else None
    return data


def load_avoid_metadata(result_dir: Path) -> dict[str, object]:
    mat_path = Path(result_dir) / f"ssm_avoid_python_results{OUTPUT_SUFFIX}.mat"
    metadata = {
        "pos_end": DEFAULT_POS_END_M,
        "pos_obstacle": np.empty((0, 3)),
        "obstacle_node_positions": np.empty((0, 3)),
        "obstacle_node_indices_1based": np.empty((0,), dtype=int),
        "obstacle_radius": DEFAULT_OBSTACLE_RADIUS_M,
        "d_safe": None,
    }
    if not mat_path.exists():
        return metadata

    mat_data = sciIO.loadmat(str(mat_path), squeeze_me=True)
    if "pos_end" in mat_data:
        metadata["pos_end"] = np.asarray(mat_data["pos_end"], dtype=float).reshape(3)
    if "pos_obstacle" in mat_data:
        metadata["pos_obstacle"] = np.asarray(mat_data["pos_obstacle"], dtype=float).reshape(-1, 3)
    if "obstacle_node_positions" in mat_data:
        metadata["obstacle_node_positions"] = np.asarray(
            mat_data["obstacle_node_positions"],
            dtype=float,
        ).reshape(-1, 3)
    if "obstacle_node_indices_1based" in mat_data:
        metadata["obstacle_node_indices_1based"] = np.asarray(
            mat_data["obstacle_node_indices_1based"],
            dtype=int,
        ).reshape(-1)
    if "obstacle_radius" in mat_data:
        metadata["obstacle_radius"] = float(np.asarray(mat_data["obstacle_radius"]).reshape(-1)[0])
    if "d_safe" in mat_data:
        metadata["d_safe"] = float(np.asarray(mat_data["d_safe"]).reshape(-1)[0])
    return metadata


def _scaled_w0(rom_data: dict):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR
    return w0


def node_xyz_from_pk(rom_data: dict, pk: np.ndarray) -> np.ndarray:
    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    ssm_state = reduced_to_full(np.asarray(pk, dtype=float).reshape(-1), _scaled_w0(rom_data), None, 0).reshape(-1)
    full_size = 3 * nodes0_xyz.shape[0]
    if ssm_state.size < full_size:
        raise ValueError(f"SSM state has {ssm_state.size} values, expected at least {full_size}.")
    displacement_xyz = ssm_state[:full_size].reshape(-1, 3)
    return nodes0_xyz + displacement_xyz


def offset_obstacles_away_from_arm(
    obstacles_m: np.ndarray,
    node_positions_m: np.ndarray,
    nodes_xyz: np.ndarray,
    offset_m: float = OBSTACLE_DISPLAY_OFFSET_M,
) -> np.ndarray:
    obstacles = np.asarray(obstacles_m, dtype=float).reshape(-1, 3)
    if obstacles.size == 0 or offset_m == 0:
        return obstacles

    node_positions = np.asarray(node_positions_m, dtype=float).reshape(-1, 3)
    arm_nodes = np.asarray(nodes_xyz, dtype=float).reshape(-1, 3)
    shifted = obstacles.copy()
    for idx, obstacle in enumerate(obstacles):
        if idx < node_positions.shape[0] and np.all(np.isfinite(node_positions[idx])):
            anchor = node_positions[idx]
        elif arm_nodes.size:
            distances = np.linalg.norm(arm_nodes - obstacle.reshape(1, 3), axis=1)
            anchor = arm_nodes[int(np.argmin(distances))]
        else:
            anchor = np.zeros(3)

        direction = obstacle - anchor
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
        shifted[idx] = obstacle + float(offset_m) * direction
    return shifted


def _plot_coordinates(
    xyz: np.ndarray,
    mirror_x_axis: bool = GEOMETRY_MIRROR_X_AXIS,
    mirror_y_axis: bool = GEOMETRY_MIRROR_Y_AXIS,
    reverse_z_axis: bool = GEOMETRY_REVERSE_Z_AXIS,
) -> np.ndarray:
    plotted = np.asarray(xyz, dtype=float).copy()
    if mirror_x_axis:
        plotted[..., 0] *= -1.0
    if mirror_y_axis:
        plotted[..., 1] *= -1.0
    if reverse_z_axis:
        plotted[..., 2] *= -1.0
    return plotted


def _sphere_mesh(center, radius, theta_count=40, phi_count=20):
    theta = np.linspace(0.0, 2.0 * np.pi, theta_count)
    phi = np.linspace(0.0, np.pi, phi_count)
    center = np.asarray(center, dtype=float).reshape(3)
    sphere_x = center[0] + radius * np.outer(np.cos(theta), np.sin(phi))
    sphere_y = center[1] + radius * np.outer(np.sin(theta), np.sin(phi))
    sphere_z = center[2] + radius * np.outer(np.ones_like(theta), np.cos(phi))
    return sphere_x, sphere_y, sphere_z


def _set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _trim_png_whitespace(path: Path, margin_px: int = GEOMETRY_CROP_MARGIN_PX) -> None:
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return

    path = Path(path)
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
            diff = ImageChops.difference(image, background)
            bbox = diff.getbbox()
            if bbox is None:
                return
            left = max(0, bbox[0] - margin_px)
            upper = max(0, bbox[1] - margin_px)
            right = min(image.width, bbox[2] + margin_px)
            lower = min(image.height, bbox[3] + margin_px)
            image.crop((left, upper, right, lower)).save(path)
    except Exception as exc:
        print(f"Whitespace trimming skipped for {path}: {exc}")


def _centerline(nodes0_xyz: np.ndarray, nodes_xyz: np.ndarray, section_count: int = 36):
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


def select_pk_samples(
    *,
    sample_indices: np.ndarray,
    result_dir: Path,
) -> list[dict[str, object]]:
    data = load_result_data(result_dir)
    pk_save = np.asarray(data["pk_save"], dtype=float)
    if pk_save.ndim != 2:
        raise ValueError(f"Expected pk_save shape (steps, pk_dim), got {pk_save.shape}")
    time_s = None if data["t"] is None else np.asarray(data["t"], dtype=float).reshape(-1)
    if time_s is not None and time_s.size != pk_save.shape[0]:
        raise ValueError(f"Time/pk_save length mismatch: t={time_s.size}, pk_save={pk_save.shape[0]}")

    sample_indices = np.asarray(sample_indices, dtype=int).reshape(-1)
    if sample_indices.size == 0:
        raise ValueError("At least one pk_save row index is required.")
    invalid = sample_indices[(sample_indices < 0) | (sample_indices >= pk_save.shape[0])]
    if invalid.size:
        raise IndexError(
            f"pk_save row index out of range: {invalid.tolist()}. "
            f"Valid range is [0, {pk_save.shape[0] - 1}]."
        )

    results = []
    for query_index, sample_index in enumerate(sample_indices):
        pk = np.asarray(pk_save[int(sample_index)], dtype=float).reshape(-1)
        if not np.all(np.isfinite(pk)):
            raise ValueError(f"pk_save row {sample_index} contains NaN or Inf.")
        item = {
            "query_index": query_index,
            "sample_index": int(sample_index),
            "time_s": float(time_s[int(sample_index)]) if time_s is not None else float(sample_index),
            "pk_est": pk,
        }
        results.append(item)
    return results


def _geometry_png_name(time_s: float, query_index: int, sample_index: int) -> str:
    safe_time = f"{time_s:.6f}".replace(".", "p").replace("-", "m")
    return f"pk_save_geometry_{query_index:03d}_step{sample_index:04d}_t{safe_time}s.png"


def _surface_from_nodes(nodes_xyz: np.ndarray, alpha: float, scalars: np.ndarray | None = None):
    import pyvista as pv

    cloud = pv.PolyData(np.asarray(nodes_xyz, dtype=float))
    if scalars is not None:
        cloud[DISPLACEMENT_SCALAR_NAME] = np.asarray(scalars, dtype=float).reshape(-1)
    try:
        surface = cloud.delaunay_3d(alpha=alpha).extract_surface(algorithm="dataset_surface").clean()
    except RuntimeError:
        surface = pv.PolyData()
    return surface if surface.n_points else cloud


def _geometry_surface(nodes_xyz: np.ndarray, *, alpha: float, scalars: np.ndarray | None = None, smooth: bool = False):
    surface = _surface_from_nodes(nodes_xyz, alpha, scalars=scalars)
    if smooth:
        try:
            surface = surface.smooth(n_iter=50, relaxation_factor=0.02, feature_angle=60.0)
        except (RuntimeError, ValueError):
            pass
    return surface


def _sphere(center: np.ndarray, radius: float):
    import pyvista as pv

    return pv.Sphere(
        radius=float(radius),
        center=np.asarray(center, dtype=float).reshape(3),
        theta_resolution=32,
        phi_resolution=16,
    )


def _camera_position_for_geometry(
    nodes0_xyz: np.ndarray,
    nodes_xyz: np.ndarray,
    extra_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center0, center = _centerline(nodes0_xyz, nodes_xyz)
    scene_points = [nodes0_xyz, nodes_xyz]
    if np.asarray(extra_points).size:
        scene_points.append(np.asarray(extra_points, dtype=float).reshape(-1, 3))
    bounds = np.vstack(scene_points)
    focal = 0.5 * (np.min(bounds, axis=0) + np.max(bounds, axis=0))
    span = np.ptp(bounds, axis=0)
    distance = 2.15 * max(float(np.linalg.norm(span)), 0.25)

    if center.shape[0] < 2:
        return focal + np.array([distance, -distance, 0.45 * distance]), focal, np.array([0.0, 0.0, 1.0])

    axial = center0[-1] - center0[0]
    axial_norm = np.linalg.norm(axial)
    axial = axial / axial_norm if axial_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    deformation = center - center0
    lateral = deformation - (deformation @ axial).reshape(-1, 1) * axial.reshape(1, 3)
    bend = lateral[int(np.argmax(np.linalg.norm(lateral, axis=1)))]
    bend_norm = np.linalg.norm(bend)
    bend = np.array([1.0, 0.0, 0.0]) if bend_norm < 1e-7 else bend / bend_norm
    view_normal = np.cross(axial, bend)
    normal_norm = np.linalg.norm(view_normal)
    view_normal = np.array([1.0, -1.0, 0.35]) if normal_norm < 1e-7 else view_normal / normal_norm
    return focal + distance * view_normal + 0.18 * distance * axial, focal, axial


def plot_geometry_pyvista(
    output_path: Path,
    rom_data: dict,
    result: dict[str, object],
    metadata: dict[str, object],
    *,
    delaunay_alpha: float,
    interactive_preview: bool,
) -> bool:
    try:
        import pyvista as pv
    except ImportError:
        return False

    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    nodes_xyz = node_xyz_from_pk(rom_data, result["pk_est"])
    displacement_mm = np.linalg.norm(nodes_xyz - nodes0_xyz, axis=1) * 1000.0
    nodes0_plot = _plot_coordinates(nodes0_xyz)
    nodes_plot = _plot_coordinates(nodes_xyz)

    obstacles_m = np.asarray(metadata.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_positions_m = np.asarray(metadata.get("obstacle_node_positions", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    display_obstacles_m = offset_obstacles_away_from_arm(obstacles_m, node_positions_m, nodes_xyz)
    obstacle_radius_m = float(metadata.get("obstacle_radius", DEFAULT_OBSTACLE_RADIUS_M))
    d_safe = metadata.get("d_safe")
    safe_radius_m = None if d_safe is None else float(d_safe)
    obstacles_plot = _plot_coordinates(display_obstacles_m) if display_obstacles_m.size else np.empty((0, 3))
    node_positions_plot = _plot_coordinates(node_positions_m) if node_positions_m.size else np.empty((0, 3))

    pv.OFF_SCREEN = not interactive_preview
    plotter = pv.Plotter(off_screen=not interactive_preview, window_size=GEOMETRY_WINDOW_SIZE)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_trackball_style()

    reference_surface = _geometry_surface(nodes0_plot, alpha=delaunay_alpha, smooth=False)
    plotter.add_mesh(
        reference_surface,
        color="#b8b8b8",
        opacity=0.13,
        smooth_shading=True,
        show_edges=False,
    )

    surface = _geometry_surface(nodes_plot, alpha=delaunay_alpha, scalars=displacement_mm, smooth=True)
    plotter.add_mesh(
        surface,
        scalars=DISPLACEMENT_SCALAR_NAME if DISPLACEMENT_SCALAR_NAME in surface.array_names else None,
        cmap="viridis",
        color="#7aa6c2",
        opacity=0.62,
        smooth_shading=True,
        show_edges=True,
        edge_color="#004f8f",
        line_width=0.3,
        show_scalar_bar=False,
        specular=0.25,
        specular_power=18,
    )

    _, centerline = _centerline(nodes0_plot, nodes_plot)
    if centerline.shape[0] > 1:
        plotter.add_lines(centerline, color="#003b66", width=4, connected=True)

    for obstacle_idx, obstacle in enumerate(obstacles_plot):
        plotter.add_mesh(_sphere(obstacle, obstacle_radius_m), color="#d62728", opacity=0.25)
        plotter.add_mesh(_sphere(obstacle, 0.004), color="#d62728", opacity=1.0)
        if safe_radius_m is not None:
            plotter.add_mesh(
                _sphere(obstacle, safe_radius_m),
                color="#d62728",
                opacity=0.08,
                style="wireframe",
                line_width=1.0,
            )
        if obstacle_idx < node_positions_plot.shape[0]:
            node = node_positions_plot[obstacle_idx]
            plotter.add_mesh(_sphere(node, 0.0045), color="#2ca02c", opacity=1.0)
            plotter.add_lines(np.vstack([obstacle, node]), color="#2ca02c", width=3, connected=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extra_points = np.vstack([obstacles_plot, node_positions_plot]) if obstacles_plot.size or node_positions_plot.size else np.empty((0, 3))
    plotter.camera_position = _camera_position_for_geometry(nodes0_plot, nodes_plot, extra_points)
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(1.22)

    if interactive_preview:
        if output_path.exists():
            output_path.unlink()
        saved = {"ok": False}

        def save_current_view_and_close() -> None:
            plotter.render()
            plotter.screenshot(str(output_path))
            if not output_path.exists():
                raise RuntimeError(f"PyVista did not save screenshot: {output_path}")
            _trim_png_whitespace(output_path)
            saved["ok"] = True
            print(f"Saved {output_path}")
            plotter.close()

        print(f"Adjust the view freely for {output_path.name}, then press 'S' to save and close.")
        plotter.add_key_event("s", save_current_view_and_close)
        plotter.add_key_event("S", save_current_view_and_close)
        plotter.show(auto_close=False, interactive=True)
        if not saved["ok"]:
            raise RuntimeError(f"No image saved. Press 'S' in the PyVista window to save and close: {output_path}")
    else:
        plotter.screenshot(str(output_path))
        plotter.close()
        if not output_path.exists():
            raise RuntimeError(f"PyVista did not save screenshot: {output_path}")
        _trim_png_whitespace(output_path)
    return True


def _plot_geometry_matplotlib(
    *,
    output_path: Path,
    rom_data: dict,
    result: dict[str, object],
    metadata: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    nodes_xyz = node_xyz_from_pk(rom_data, result["pk_est"])
    nodes0_plot = _plot_coordinates(nodes0_xyz)
    nodes_plot = _plot_coordinates(nodes_xyz)
    center0, center = _centerline(nodes0_plot, nodes_plot)
    obstacles_m = np.asarray(metadata.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_positions_m = np.asarray(metadata.get("obstacle_node_positions", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    display_obstacles_m = offset_obstacles_away_from_arm(obstacles_m, node_positions_m, nodes_xyz)
    obstacles_plot = _plot_coordinates(display_obstacles_m) if display_obstacles_m.size else np.empty((0, 3))
    node_positions_plot = _plot_coordinates(node_positions_m) if node_positions_m.size else np.empty((0, 3))
    obstacle_radius_m = float(metadata.get("obstacle_radius", DEFAULT_OBSTACLE_RADIUS_M))
    d_safe = metadata.get("d_safe")
    safe_radius_m = None if d_safe is None else float(d_safe)

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(nodes0_plot[:, 0], nodes0_plot[:, 1], nodes0_plot[:, 2], c="#b8b8b8", s=6, alpha=0.18, label="reference nodes")
    ax.scatter(nodes_plot[:, 0], nodes_plot[:, 1], nodes_plot[:, 2], c="#1f77b4", s=8, alpha=0.55, label="estimated nodes")
    if center0.shape[0] > 1:
        ax.plot(center0[:, 0], center0[:, 1], center0[:, 2], color="#777777", linestyle=":", linewidth=2, label="reference centerline")
    if center.shape[0] > 1:
        ax.plot(center[:, 0], center[:, 1], center[:, 2], color="#003b66", linewidth=3, label="estimated centerline")

    for obstacle_idx, obstacle_m in enumerate(obstacles_plot):
        ax.scatter(obstacle_m[0], obstacle_m[1], obstacle_m[2], c="r", marker="x", s=90, label=f"obstacle {obstacle_idx + 1}")
        sx, sy, sz = _sphere_mesh(obstacle_m, obstacle_radius_m)
        ax.plot_surface(sx, sy, sz, color="tab:red", alpha=0.2, linewidth=0)
        if safe_radius_m is not None:
            sx, sy, sz = _sphere_mesh(obstacle_m, safe_radius_m, theta_count=32, phi_count=16)
            ax.plot_wireframe(sx, sy, sz, color="r", linestyle=":", linewidth=0.6)
        if obstacle_idx < node_positions_plot.shape[0]:
            node = node_positions_plot[obstacle_idx]
            ax.scatter(node[0], node[1], node[2], c="tab:green", marker="o", s=45, label="obstacle node")
            ax.plot([obstacle_m[0], node[0]], [obstacle_m[1], node[1]], [obstacle_m[2], node[2]], color="tab:green", linestyle=":")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    _set_axes_equal_3d(ax)
    ax.view_init(elev=27, azim=-69, roll=0)
    ax.set_axis_off()
    ax.margins(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    _trim_png_whitespace(output_path)


def save_geometry_figures(
    *,
    results: list[dict[str, object]],
    result_dir: Path,
    trajectory_name: str,
    figure_dir: Path,
    alpha: float,
    interactive_view: bool,
) -> list[Path]:
    rom_data = load_rom_data(MODEL_INFO_DIR, traj_dir=resolve_trajectory_dir(trajectory_name))
    metadata = load_avoid_metadata(result_dir)
    output_paths = []
    for result in results:
        output_path = figure_dir / _geometry_png_name(
            float(result["time_s"]),
            int(result["query_index"]),
            int(result["sample_index"]),
        )
        try:
            plotted = plot_geometry_pyvista(
                output_path,
                rom_data,
                result,
                metadata,
                delaunay_alpha=alpha,
                interactive_preview=interactive_view,
            )
        except Exception as exc:
            if interactive_view:
                raise
            print(f"PyVista geometry plot failed at t={result['time_s']:.8g}s; using Matplotlib fallback: {exc}")
            plotted = False
        if not plotted:
            _plot_geometry_matplotlib(
                output_path=output_path,
                rom_data=rom_data,
                result=result,
                metadata=metadata,
            )
        result["geometry_figure"] = str(output_path)
        output_paths.append(output_path)
    return output_paths


def parse_args() -> argparse.Namespace:
    output_dir = SCRIPT_DIR / "pk_save_geometry"
    parser = argparse.ArgumentParser(
        description="Plot SSM geometry from user-selected rows in pk_save_ssm_avoid_complex.npy.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=USER_DEFINED_PK_INDICES,
        help=(
            "Zero-based row indices in pk_save_ssm_avoid_complex.npy. "
            "If omitted, USER_DEFINED_PK_INDICES at the top of this script is used."
        ),
    )
    parser.add_argument("--result-dir", type=Path, default=SCRIPT_DIR, help="Directory containing avoid result .npy files.")
    parser.add_argument("--trajectory", choices=("eight", "spiral"), default=TRAJECTORY_NAME)
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=output_dir / GEOMETRY_FIGURE_DIR_NAME,
        help="Directory for SSM geometry figures with obstacles.",
    )
    parser.add_argument("--geometry-alpha", type=float, default=0.035, help="PyVista alpha value for surface reconstruction.")
    geometry_view_group = parser.add_mutually_exclusive_group()
    geometry_view_group.add_argument(
        "--interactive-geometry",
        dest="interactive_geometry",
        action="store_true",
        default=GEOMETRY_INTERACTIVE_VIEW,
        help="Open an interactive geometry window; press 'S' to save the current view and close. This is the default.",
    )
    geometry_view_group.add_argument(
        "--offscreen-geometry",
        dest="interactive_geometry",
        action="store_false",
        help="Save geometry figures directly without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_indices = np.asarray(args.indices, dtype=int)
    results = select_pk_samples(
        sample_indices=sample_indices,
        result_dir=args.result_dir,
    )
    geometry_paths = save_geometry_figures(
        results=results,
        result_dir=args.result_dir,
        trajectory_name=args.trajectory,
        figure_dir=args.geometry_dir,
        alpha=args.geometry_alpha,
        interactive_view=args.interactive_geometry,
    )
    print(f"Plotted geometry for {len(results)} pk_save row(s): {sample_indices.tolist()}")
    print(f"Saved geometry figures: {args.geometry_dir}")
    for path in geometry_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
