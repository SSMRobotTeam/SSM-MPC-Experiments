from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamics.SSM.matlab_data import ROM_DIR, clone_struct_array, get_field, load_rom_data, loadmat
from dynamics.SSM.ssm_dynamics import reduced_to_full


PLOT_DIR = Path(__file__).resolve().parent
PLAN_RESULT_DIR = PLOT_DIR / "Plan_And_Avoid"
STATIC_VERIFICATION_DIR = REPO_ROOT / "StaticVerification"
SUFFIX = "_plan_avoid_complex"
RESULT_MAT = PLAN_RESULT_DIR / f"ssm_python_results{SUFFIX}.mat"
DEFAULT_POS_END_M = np.array([0.0, 0.0, 0.4400])
DEFAULT_POS_MID_M = np.array([0.0, 0.0, 0.2225])
DEFAULT_POS_BASE_M = np.array([0.0, 0.0, 0.0])
DEFAULT_OBSTACLE_RADIUS_M = 0.015
SSM_FACTOR = 1.475
DISPLACEMENT_SCALAR_NAME = "displacement_mm"


def _load_optional_npy(plot_dir: Path, name: str):
    path = plot_dir / f"{name}{SUFFIX}.npy"
    return np.load(path) if path.exists() else None


def load_track_and_plan_results(plot_dir: str | Path = PLAN_RESULT_DIR):
    plot_dir = Path(plot_dir)
    mat_path = plot_dir / f"ssm_python_results{SUFFIX}.mat"
    results = {
        "t": _load_optional_npy(plot_dir, "t_ssm"),
        "y_exprm": _load_optional_npy(plot_dir, "y_exprm_ssm"),
        "y_ssm": _load_optional_npy(plot_dir, "y_ssm"),
        "ut_save": _load_optional_npy(plot_dir, "ut_save_ssm"),
        "pk_save": _load_optional_npy(plot_dir, "pk_save_ssm"),
        "target_error_mm": _load_optional_npy(plot_dir, "target_error_ssm"),
        "planned_zd": _load_optional_npy(plot_dir, "planned_zd_ssm"),
        "planner_waypoints_m": _load_optional_npy(plot_dir, "planner_waypoints_ssm"),
        "pos_end": DEFAULT_POS_END_M,
        "pos_obstacle": _load_optional_npy(plot_dir, "obstacle_positions_ssm"),
        "obstacle_node_positions": np.empty((0, 3)),
        "obstacle_node_indices_1based": np.empty((0,), dtype=int),
        "obstacle_radius": DEFAULT_OBSTACLE_RADIUS_M,
        "d_safe": None,
        "target_displacement_mm": None,
    }

    if mat_path.exists():
        mat_data = sciIO.loadmat(str(mat_path), squeeze_me=True)
        for key in (
            "t",
            "y_exprm",
            "y_ssm",
            "pk_save",
            "ut_save",
            "target_error_mm",
            "planned_zd",
            "planner_waypoints_m",
        ):
            if key in mat_data:
                results[key] = np.asarray(mat_data[key], dtype=float)
        if "pos_end" in mat_data:
            results["pos_end"] = np.asarray(mat_data["pos_end"], dtype=float).reshape(3)
        if "pos_obstacle" in mat_data:
            results["pos_obstacle"] = np.asarray(mat_data["pos_obstacle"], dtype=float).reshape(-1, 3)
        if "obstacle_node_positions" in mat_data:
            results["obstacle_node_positions"] = np.asarray(
                mat_data["obstacle_node_positions"],
                dtype=float,
            ).reshape(-1, 3)
        if "obstacle_node_indices_1based" in mat_data:
            results["obstacle_node_indices_1based"] = np.asarray(
                mat_data["obstacle_node_indices_1based"],
                dtype=int,
            ).reshape(-1)
        if "obstacle_radius" in mat_data:
            results["obstacle_radius"] = float(np.asarray(mat_data["obstacle_radius"]).reshape(-1)[0])
        if "d_safe" in mat_data:
            results["d_safe"] = float(np.asarray(mat_data["d_safe"]).reshape(-1)[0])
        if "target_displacement_mm" in mat_data:
            results["target_displacement_mm"] = np.asarray(mat_data["target_displacement_mm"], dtype=float).reshape(3)

    if results["pos_obstacle"] is None:
        results["pos_obstacle"] = np.empty((0, 3))

    required = ("t", "y_exprm", "ut_save", "planned_zd")
    missing = [key for key in required if results[key] is None]
    if missing:
        raise FileNotFoundError(
            f"Missing plan/avoid result data: {', '.join(missing)}. "
            f"Run Main_demo_ssm_PlanAndAvoid_complex.py first, or check {mat_path}."
        )

    results["t"] = np.asarray(results["t"], dtype=float).reshape(-1)
    results["y_exprm"] = np.asarray(results["y_exprm"], dtype=float)
    results["ut_save"] = np.asarray(results["ut_save"], dtype=float)
    results["planned_zd"] = _as_3_by_n(results["planned_zd"])
    if results["pk_save"] is not None:
        results["pk_save"] = np.asarray(results["pk_save"], dtype=float)
    if results["y_ssm"] is not None:
        results["y_ssm"] = np.asarray(results["y_ssm"], dtype=float)
    return results


def _as_3_by_n(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected planned_zd to be a 2-D array, got shape {values.shape}")
    if values.shape[0] >= 3:
        return values[:3, :]
    if values.shape[1] >= 3:
        return values[:, :3].T
    raise ValueError(f"Expected planned_zd to contain xyz rows or columns, got shape {values.shape}")


def _set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _sphere_mesh(center_mm, radius_mm, theta_count=40, phi_count=20):
    theta = np.linspace(0.0, 2.0 * np.pi, theta_count)
    phi = np.linspace(0.0, np.pi, phi_count)
    sphere_x = center_mm[0] + radius_mm * np.outer(np.cos(theta), np.sin(phi))
    sphere_y = center_mm[1] + radius_mm * np.outer(np.sin(theta), np.sin(phi))
    sphere_z = center_mm[2] + radius_mm * np.outer(np.ones_like(theta), np.cos(phi))
    return sphere_x, sphere_y, sphere_z


def _scaled_w0(rom_data: dict):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR
    return w0


def _load_tet_connect(path: Path | None):
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


def _node_xyz_from_pk(rom_data: dict, pk: np.ndarray) -> np.ndarray:
    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    ssm_state = reduced_to_full(np.asarray(pk, dtype=float).reshape(-1), _scaled_w0(rom_data), None, 0).reshape(-1)
    full_size = 3 * nodes0_xyz.shape[0]
    if ssm_state.size < full_size:
        raise ValueError(f"SSM state has {ssm_state.size} values, expected at least {full_size} displacement DOFs.")
    displacement_xyz = ssm_state[:full_size].reshape(-1, 3)
    return nodes0_xyz + displacement_xyz


def _extract_surface_from_tets(points_xyz: np.ndarray, tets: np.ndarray):
    import pyvista as pv

    faces = np.vstack(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ]
    )
    faces_sorted = np.sort(faces, axis=1)
    _, unique_indices, counts = np.unique(faces_sorted, axis=0, return_index=True, return_counts=True)
    boundary_faces = faces[unique_indices[counts == 1]]
    faces_pv = np.hstack(
        [np.full((boundary_faces.shape[0], 1), 3, dtype=np.int64), boundary_faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(np.asarray(points_xyz, dtype=float), faces_pv)


def _surface_from_nodes(nodes_xyz: np.ndarray, alpha: float, scalars: np.ndarray | None = None):
    import pyvista as pv

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
):
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


def _plot_coordinates(xyz: np.ndarray) -> np.ndarray:
    plotted = np.asarray(xyz, dtype=float).copy()
    plotted[..., 2] *= -1.0
    return plotted


def _pv_sphere(center: np.ndarray, radius: float):
    import pyvista as pv

    return pv.Sphere(radius=radius, center=np.asarray(center, dtype=float), theta_resolution=32, phi_resolution=16)


def _pv_line(point_a: np.ndarray, point_b: np.ndarray):
    import pyvista as pv

    return pv.Line(np.asarray(point_a, dtype=float), np.asarray(point_b, dtype=float))


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


def _camera_position_for_geometry(nodes0_xyz: np.ndarray, nodes_xyz: np.ndarray, extra_points: np.ndarray):
    center0, center = _centerline(nodes0_xyz, nodes_xyz)
    scene_points = np.vstack([nodes0_xyz, nodes_xyz, extra_points.reshape(-1, 3)])
    focal = 0.5 * (np.min(scene_points, axis=0) + np.max(scene_points, axis=0))
    span = np.ptp(scene_points, axis=0)
    distance = 2.2 * max(float(np.linalg.norm(span)), 0.25)

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
    bend = np.array([1.0, 0.0, 0.0]) if bend_norm < 1e-7 else bend / bend_norm
    view_normal = np.cross(axial, bend)
    normal_norm = np.linalg.norm(view_normal)
    view_normal = np.array([1.0, -1.0, 0.35]) if normal_norm < 1e-7 else view_normal / normal_norm
    camera = focal + distance * view_normal + 0.2 * distance * axial
    return camera, focal, axial


def _trim_count(t, *arrays):
    return min([t.size, *[array.shape[0] for array in arrays]])


def plot_control_trajectory(results):
    t = results["t"]
    y = results["y_exprm"]
    y_ssm = results.get("y_ssm")
    planned_zd = results["planned_zd"]
    n = min(t.size, y.shape[0], planned_zd.shape[1])

    fig = plt.figure("Track and plan trajectory")
    labels = ["end x", "end y", "end z"]
    for idx, label in enumerate(labels):
        ax = fig.add_subplot(3, 1, idx + 1)
        ax.plot(t[:n], y[:n, idx], label=f"measured {label}")
        if y_ssm is not None:
            ssm_n = min(n, y_ssm.shape[0])
            ax.plot(t[:ssm_n], y_ssm[:ssm_n, idx], label=f"SSM {label}")
        ax.plot(t[:n], planned_zd[idx, :n] * 1000.0, "k--", label=f"planned {label}")
        ax.set_ylabel("Position (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes = fig.axes
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()


def plot_target_error(results):
    target_error = results.get("target_error_mm")
    if target_error is None:
        return
    t = results["t"]
    n = min(t.size, target_error.size)

    fig = plt.figure("Track and plan target error")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t[:n], target_error[:n], label="target error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def plot_control_inputs(results):
    t = results["t"]
    u = results["ut_save"]
    n = _trim_count(t, u)

    fig = plt.figure("Track and plan control inputs")
    ax = fig.add_subplot(1, 1, 1)
    styles = ["r-", "g-", "b-", "r--", "g--", "b--"]
    labels = ["long 1", "long 2", "long 3", "short 1", "short 2", "short 3"]
    for idx in range(min(u.shape[1], len(labels))):
        ax.plot(t[:n], u[:n, idx], styles[idx], label=labels[idx])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tendon command (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def plot_3d_track_plan_obstacles(results):
    t = results["t"]
    y = results["y_exprm"]
    y_ssm = results.get("y_ssm")
    planned_zd = results["planned_zd"]
    planner_waypoints = results.get("planner_waypoints_m")
    pos_end_mm = np.asarray(results.get("pos_end", DEFAULT_POS_END_M), dtype=float).reshape(3) * 1000.0
    obstacles_m = np.asarray(results.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_positions_m = np.asarray(results.get("obstacle_node_positions", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_indices = np.asarray(results.get("obstacle_node_indices_1based", []), dtype=int).reshape(-1)
    obstacle_radius_mm = float(results.get("obstacle_radius", DEFAULT_OBSTACLE_RADIUS_M)) * 1000.0
    d_safe = results.get("d_safe")
    n = min(t.size, y.shape[0], planned_zd.shape[1])

    measured_end_mm = y[:n, :3] + pos_end_mm.reshape(1, 3)
    planned_end_mm = planned_zd[:3, :n].T * 1000.0 + pos_end_mm.reshape(1, 3)

    fig = plt.figure("Track and plan 3D obstacles")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(measured_end_mm[:, 0], measured_end_mm[:, 1], measured_end_mm[:, 2], label="measured end")
    ax.plot(planned_end_mm[:, 0], planned_end_mm[:, 1], planned_end_mm[:, 2], "k--", label="planned end")
    if planner_waypoints is not None:
        waypoint_mm = np.asarray(planner_waypoints, dtype=float).reshape(-1, 3) * 1000.0
        ax.scatter(
            waypoint_mm[:, 0],
            waypoint_mm[:, 1],
            waypoint_mm[:, 2],
            c="k",
            marker="^",
            s=45,
            label="planner waypoints",
        )

    if y_ssm is not None:
        ssm_n = min(n, y_ssm.shape[0])
        ssm_end_mm = y_ssm[:ssm_n, :3] + pos_end_mm.reshape(1, 3)
        ax.plot(ssm_end_mm[:, 0], ssm_end_mm[:, 1], ssm_end_mm[:, 2], label="SSM end")

    safe_radius_mm = None if d_safe is None else float(d_safe) * 1000.0
    for obstacle_idx, obstacle_m in enumerate(obstacles_m):
        obstacle_mm = obstacle_m * 1000.0
        ax.scatter(
            obstacle_mm[0],
            obstacle_mm[1],
            obstacle_mm[2],
            c="r",
            marker="x",
            s=85,
            label=f"obstacle {obstacle_idx + 1}",
        )
        sphere_x, sphere_y, sphere_z = _sphere_mesh(obstacle_mm, obstacle_radius_mm)
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color="tab:red", alpha=0.18, linewidth=0)

        if safe_radius_mm is not None:
            safe_x, safe_y, safe_z = _sphere_mesh(obstacle_mm, safe_radius_mm, theta_count=32, phi_count=16)
            ax.plot_wireframe(safe_x, safe_y, safe_z, color="r", linestyle=":", linewidth=0.6)

        if obstacle_idx < node_positions_m.shape[0]:
            node_mm = node_positions_m[obstacle_idx] * 1000.0
            node_label = f"nearest node {node_indices[obstacle_idx]}" if obstacle_idx < node_indices.size else "nearest node"
            ax.scatter(node_mm[0], node_mm[1], node_mm[2], c="tab:green", marker="o", s=45, label=node_label)
            ax.plot(
                [obstacle_mm[0], node_mm[0]],
                [obstacle_mm[1], node_mm[1]],
                [obstacle_mm[2], node_mm[2]],
                color="tab:green",
                linestyle=":",
                linewidth=0.8,
            )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    _set_axes_equal_3d(ax)
    ax.legend()
    fig.tight_layout()


def plot_final_structure_obstacles(results):
    if results.get("pk_save") is not None:
        try:
            plot_final_ssm_geometry_obstacles(results)
            return
        except ImportError as exc:
            print(f"PyVista geometry plot skipped because dependency is missing: {exc}")
        except Exception as exc:
            print(f"PyVista geometry plot failed, falling back to simplified structure plot: {exc}")

    t = results["t"]
    y = results["y_exprm"]
    y_ssm = results.get("y_ssm")
    pos_end_m = np.asarray(results.get("pos_end", DEFAULT_POS_END_M), dtype=float).reshape(3)
    pos_mid_m = np.asarray(results.get("pos_mid", DEFAULT_POS_MID_M), dtype=float).reshape(3)
    pos_base_m = np.asarray(results.get("pos_base", DEFAULT_POS_BASE_M), dtype=float).reshape(3)
    obstacles_m = np.asarray(results.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_positions_m = np.asarray(results.get("obstacle_node_positions", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_indices = np.asarray(results.get("obstacle_node_indices_1based", []), dtype=int).reshape(-1)
    obstacle_radius_mm = float(results.get("obstacle_radius", DEFAULT_OBSTACLE_RADIUS_M)) * 1000.0
    d_safe = results.get("d_safe")
    n = min(t.size, y.shape[0])
    if n == 0:
        return

    final_y = y[n - 1]
    final_base_mm = pos_base_m * 1000.0
    final_mid_mm = pos_mid_m * 1000.0 + final_y[3:6]
    final_end_mm = pos_end_m * 1000.0 + final_y[:3]
    measured_structure = np.vstack([final_base_mm, final_mid_mm, final_end_mm])

    fig = plt.figure("Final 3D structure and obstacles")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(
        measured_structure[:, 0],
        measured_structure[:, 1],
        measured_structure[:, 2],
        "-o",
        color="tab:blue",
        linewidth=2.0,
        markersize=5,
        label="measured final structure",
    )

    if y_ssm is not None and y_ssm.shape[0] > 0:
        ssm_idx = min(n - 1, y_ssm.shape[0] - 1)
        ssm_mid_mm = pos_mid_m * 1000.0 + y_ssm[ssm_idx, 3:6]
        ssm_end_mm = pos_end_m * 1000.0 + y_ssm[ssm_idx, :3]
        ssm_structure = np.vstack([final_base_mm, ssm_mid_mm, ssm_end_mm])
        ax.plot(
            ssm_structure[:, 0],
            ssm_structure[:, 1],
            ssm_structure[:, 2],
            "--o",
            color="tab:orange",
            linewidth=1.6,
            markersize=4,
            label="SSM final structure",
        )

    if final_y.size >= 9:
        feedback_mm = pos_mid_m * 1000.0 + final_y[6:9]
        ax.scatter(
            feedback_mm[0],
            feedback_mm[1],
            feedback_mm[2],
            c="tab:purple",
            marker="s",
            s=45,
            label="final feedback point",
        )

    safe_radius_mm = None if d_safe is None else float(d_safe) * 1000.0
    for obstacle_idx, obstacle_m in enumerate(obstacles_m):
        obstacle_mm = obstacle_m * 1000.0
        ax.scatter(
            obstacle_mm[0],
            obstacle_mm[1],
            obstacle_mm[2],
            c="r",
            marker="x",
            s=90,
            label=f"obstacle {obstacle_idx + 1}",
        )
        sphere_x, sphere_y, sphere_z = _sphere_mesh(obstacle_mm, obstacle_radius_mm)
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color="tab:red", alpha=0.2, linewidth=0)

        if safe_radius_mm is not None:
            safe_x, safe_y, safe_z = _sphere_mesh(obstacle_mm, safe_radius_mm, theta_count=32, phi_count=16)
            ax.plot_wireframe(safe_x, safe_y, safe_z, color="r", linestyle=":", linewidth=0.6)

        if obstacle_idx < node_positions_m.shape[0]:
            node_mm = node_positions_m[obstacle_idx] * 1000.0
            node_label = f"nearest node {node_indices[obstacle_idx]}" if obstacle_idx < node_indices.size else "nearest node"
            ax.scatter(node_mm[0], node_mm[1], node_mm[2], c="tab:green", marker="o", s=45, label=node_label)
            ax.plot(
                [obstacle_mm[0], node_mm[0]],
                [obstacle_mm[1], node_mm[1]],
                [obstacle_mm[2], node_mm[2]],
                color="tab:green",
                linestyle=":",
                linewidth=0.8,
            )

    ax.set_title(f"Final structure at t={t[n - 1]:.3f} s")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    _set_axes_equal_3d(ax)
    ax.legend()
    fig.tight_layout()


def plot_final_ssm_geometry_obstacles(results):
    import pyvista as pv

    t = results["t"]
    pk_save = np.asarray(results["pk_save"], dtype=float)
    if pk_save.shape[0] == 0:
        return

    final_idx = min(t.size, pk_save.shape[0]) - 1
    rom_data = load_rom_data(ROM_DIR)
    tet_connect = _load_tet_connect(STATIC_VERIFICATION_DIR / "tet_connect.mat")
    nodes0_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    nodes_xyz = _node_xyz_from_pk(rom_data, pk_save[final_idx])
    displacement_mm = np.linalg.norm(nodes_xyz - nodes0_xyz, axis=1) * 1000.0

    obstacles_m = np.asarray(results.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_positions_m = np.asarray(results.get("obstacle_node_positions", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_indices = np.asarray(results.get("obstacle_node_indices_1based", []), dtype=int).reshape(-1)
    obstacle_radius_m = float(results.get("obstacle_radius", DEFAULT_OBSTACLE_RADIUS_M))
    d_safe = results.get("d_safe")
    safe_radius_m = None if d_safe is None else float(d_safe)

    nodes0_plot = _plot_coordinates(nodes0_xyz)
    nodes_plot = _plot_coordinates(nodes_xyz)
    obstacles_plot = _plot_coordinates(obstacles_m) if obstacles_m.size else np.empty((0, 3))
    node_positions_plot = _plot_coordinates(node_positions_m) if node_positions_m.size else np.empty((0, 3))
    extra_points = np.vstack([nodes_plot, obstacles_plot, node_positions_plot])

    plotter = pv.Plotter(window_size=(1600, 1200))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_trackball_style()

    reference_surface = _geometry_surface(
        nodes0_plot,
        alpha=0.035,
        tet_connect=tet_connect,
        smooth=False,
    )
    plotter.add_mesh(
        reference_surface,
        color="#b8b8b8",
        opacity=0.14,
        smooth_shading=True,
        show_edges=False,
    )

    surface = _geometry_surface(
        nodes_plot,
        alpha=0.035,
        tet_connect=tet_connect,
        scalars=displacement_mm,
        smooth=True,
    )
    plotter.add_mesh(
        surface,
        scalars=DISPLACEMENT_SCALAR_NAME if DISPLACEMENT_SCALAR_NAME in surface.array_names else None,
        cmap="viridis",
        color="#7aa6c2",
        opacity=0.55,
        smooth_shading=True,
        show_edges=True,
        edge_color="#004f8f",
        line_width=0.25,
        specular=0.25,
        specular_power=18,
        scalar_bar_args={"title": "disp (mm)"},
    )

    for obstacle_idx, obstacle in enumerate(obstacles_plot):
        plotter.add_mesh(_pv_sphere(obstacle, obstacle_radius_m), color="#d62728", opacity=0.25)
        plotter.add_mesh(_pv_sphere(obstacle, 0.004), color="#d62728", opacity=1.0)
        plotter.add_point_labels(
            obstacle.reshape(1, 3),
            [f"obstacle {obstacle_idx + 1}"],
            font_size=12,
            text_color="#7f0000",
            point_color="#d62728",
            point_size=8,
            shape_opacity=0.15,
        )
        if safe_radius_m is not None:
            plotter.add_mesh(
                _pv_sphere(obstacle, safe_radius_m),
                color="#d62728",
                opacity=0.08,
                style="wireframe",
                line_width=1.0,
            )
        if obstacle_idx < node_positions_plot.shape[0]:
            node = node_positions_plot[obstacle_idx]
            label = f"node {node_indices[obstacle_idx]}" if obstacle_idx < node_indices.size else "nearest node"
            plotter.add_mesh(_pv_sphere(node, 0.0045), color="#2ca02c", opacity=1.0)
            plotter.add_mesh(_pv_line(obstacle, node), color="#2ca02c", line_width=2)
            plotter.add_point_labels(
                node.reshape(1, 3),
                [label],
                font_size=11,
                text_color="#145214",
                point_color="#2ca02c",
                point_size=7,
                shape_opacity=0.15,
            )

    plotter.add_title(f"Final SSM geometry at t={t[final_idx]:.3f} s", font_size=14)
    plotter.add_axes()
    plotter.camera_position = _camera_position_for_geometry(nodes0_plot, nodes_plot, extra_points)
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(1.12)
    plotter.show()


def main():
    results = load_track_and_plan_results()
    plot_control_trajectory(results)
    plot_target_error(results)
    plot_control_inputs(results)
    plot_3d_track_plan_obstacles(results)
    plot_final_structure_obstacles(results)
    plt.show()


if __name__ == "__main__":
    main()
