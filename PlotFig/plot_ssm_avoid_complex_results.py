from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


PLOT_DIR = Path(__file__).resolve().parent
BASE_DIR = PLOT_DIR.parent
TRAJ_DIR = BASE_DIR / "dynamics" / "trajinfom" / "eight"

SUFFIX = "_complex"
AVOID_RESULTS_MAT = PLOT_DIR / f"ssm_avoid_python_results{SUFFIX}.mat"
END_INITIAL_POSITION_M = np.array([0.0, 0.0, 0.4400])
DEFAULT_OBSTACLE_RADIUS_M = 0.02


def load_ssm_avoid_complex_results(plot_dir: str | Path = PLOT_DIR):
    plot_dir = Path(plot_dir)
    results = {
        "t": np.load(plot_dir / f"t_ssm_avoid{SUFFIX}.npy"),
        "y_exprm": np.load(plot_dir / f"y_exprm_ssm_avoid{SUFFIX}.npy"),
        "y_ssm": None,
        "ut_save": np.load(plot_dir / f"ut_save_ssm_avoid{SUFFIX}.npy"),
        "zd": sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"],
        "pos_end": END_INITIAL_POSITION_M,
        "pos_obstacle": np.empty((0, 3)),
        "obstacle_node_positions": np.empty((0, 3)),
        "obstacle_node_indices_1based": np.empty((0,), dtype=int),
        "obstacle_radius": DEFAULT_OBSTACLE_RADIUS_M,
        "d_safe": None,
    }

    y_ssm_path = plot_dir / f"y_ssm_avoid{SUFFIX}.npy"
    if y_ssm_path.exists():
        results["y_ssm"] = np.load(y_ssm_path)

    if AVOID_RESULTS_MAT.exists():
        mat_data = sciIO.loadmat(str(AVOID_RESULTS_MAT), squeeze_me=True)
        if "pos_end" in mat_data:
            results["pos_end"] = np.asarray(mat_data["pos_end"], dtype=float).reshape(3)
        if "y_ssm" in mat_data:
            results["y_ssm"] = np.asarray(mat_data["y_ssm"], dtype=float)
        if "zzd" in mat_data:
            results["zd"] = np.asarray(mat_data["zzd"], dtype=float)
        if "pos_obstacle" in mat_data:
            results["pos_obstacle"] = np.asarray(mat_data["pos_obstacle"], dtype=float).reshape(-1, 3)
        if "obstacle_node_positions" in mat_data:
            results["obstacle_node_positions"] = np.asarray(mat_data["obstacle_node_positions"], dtype=float).reshape(-1, 3)
        if "obstacle_node_indices_1based" in mat_data:
            results["obstacle_node_indices_1based"] = np.asarray(
                mat_data["obstacle_node_indices_1based"],
                dtype=int,
            ).reshape(-1)
        if "obstacle_radius" in mat_data:
            results["obstacle_radius"] = float(np.asarray(mat_data["obstacle_radius"]).reshape(-1)[0])
        d_safe = mat_data.get("d_safe")
        results["d_safe"] = None if d_safe is None else float(np.asarray(d_safe).reshape(-1)[0])

    return results


def _trim_time_series(t, *arrays):
    count = min([t.size, *[array.shape[0] for array in arrays]])
    return (t[:count], *[array[:count] for array in arrays])


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


def plot_tracking_results(results):
    t = results["t"]
    y = results["y_exprm"]
    zd = results.get("zd")
    n = min(t.size, y.shape[0])

    fig = plt.figure("SSM complex avoid tracking")
    labels = ["end x", "end y", "end z", "mid x", "mid y", "mid z"]
    for idx, label in enumerate(labels):
        ax = fig.add_subplot(3, 2, idx + 1)
        ax.plot(t[:n], y[:n, idx], label=label)
        if zd is not None and idx < 3:
            ref_n = min(n, zd.shape[1])
            ax.plot(t[:ref_n], zd[idx, :ref_n] * 1000.0, "k--", label=f"desired {label}")
        ax.set_ylabel("Position (mm)")
        ax.legend()
    fig.axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()


def plot_motor_commands(results):
    t, u = _trim_time_series(results["t"], results["ut_save"])

    fig = plt.figure("SSM complex avoid motor commands")
    ax = fig.add_subplot(1, 1, 1)
    styles = ["r-", "g-", "b-", "r--", "g--", "b--"]
    for idx in range(u.shape[1]):
        ax.plot(t, u[:, idx], styles[idx % len(styles)], label=f"u{idx + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor command (mm)")
    ax.legend()
    fig.tight_layout()


def plot_obstacle_view(results):
    t, y = _trim_time_series(results["t"], results["y_exprm"])
    y_ssm = results.get("y_ssm")
    zd = results.get("zd")
    pos_end_mm = np.asarray(results.get("pos_end", END_INITIAL_POSITION_M), dtype=float).reshape(3) * 1000.0
    obstacles_m = np.asarray(results.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_positions_m = np.asarray(results.get("obstacle_node_positions", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_indices = np.asarray(results.get("obstacle_node_indices_1based", []), dtype=int).reshape(-1)
    obstacle_radius_mm = float(results.get("obstacle_radius", DEFAULT_OBSTACLE_RADIUS_M)) * 1000.0
    d_safe = results.get("d_safe")
    measured_end_mm = y[:, :3] + pos_end_mm

    fig = plt.figure("SSM complex avoid obstacle view")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(
        measured_end_mm[:, 0],
        measured_end_mm[:, 1],
        measured_end_mm[:, 2],
        color="tab:blue",
        label="measured end",
    )

    if y_ssm is not None:
        count = min(t.size, y_ssm.shape[0])
        algorithm_end_mm = y_ssm[:count, :3] + pos_end_mm
        ax.plot(
            algorithm_end_mm[:, 0],
            algorithm_end_mm[:, 1],
            algorithm_end_mm[:, 2],
            color="tab:orange",
            label="SSM calculated end",
        )

    if zd is not None:
        ref_n = min(t.size, zd.shape[1])
        desired_end_mm = zd[:3, :ref_n].T * 1000.0 + pos_end_mm
        ax.plot(
            desired_end_mm[:, 0],
            desired_end_mm[:, 1],
            desired_end_mm[:, 2],
            "k--",
            label="desired end",
        )

    safe_radius_mm = None if d_safe is None else float(d_safe) * 1000.0
    for obstacle_idx, obstacle_m in enumerate(obstacles_m):
        obstacle_mm = obstacle_m * 1000.0
        label_suffix = f" {obstacle_idx + 1}"
        ax.scatter(
            obstacle_mm[0],
            obstacle_mm[1],
            obstacle_mm[2],
            c="r",
            marker="x",
            s=80,
            label=f"obstacle{label_suffix}",
        )
        obstacle_x, obstacle_y, obstacle_z = _sphere_mesh(obstacle_mm, obstacle_radius_mm)
        ax.plot_surface(obstacle_x, obstacle_y, obstacle_z, color="tab:red", alpha=0.2, linewidth=0)

        if safe_radius_mm is not None:
            sphere_x, sphere_y, sphere_z = _sphere_mesh(obstacle_mm, safe_radius_mm, theta_count=32, phi_count=16)
            ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r", linestyle=":", linewidth=0.6)

        if obstacle_idx < node_positions_m.shape[0]:
            node_mm = node_positions_m[obstacle_idx] * 1000.0
            node_label = f"nearest node {node_indices[obstacle_idx]}" if obstacle_idx < node_indices.size else "nearest node"
            ax.scatter(
                node_mm[0],
                node_mm[1],
                node_mm[2],
                c="tab:green",
                marker="o",
                s=45,
                label=node_label,
            )
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


def plot_obstacle_distances(results):
    obstacles_m = np.asarray(results.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    node_positions_m = np.asarray(results.get("obstacle_node_positions", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    d_safe = results.get("d_safe")
    if obstacles_m.size == 0 or node_positions_m.size == 0:
        return

    count = min(obstacles_m.shape[0], node_positions_m.shape[0])
    initial_dist_mm = np.linalg.norm(node_positions_m[:count] - obstacles_m[:count], axis=1) * 1000.0
    labels = [f"obs {idx + 1}" for idx in range(count)]

    fig = plt.figure("SSM complex avoid initial node distances")
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, initial_dist_mm, color="tab:green", alpha=0.75)
    if d_safe is not None:
        ax.axhline(float(d_safe) * 1000.0, color="r", linestyle="--", label="safe distance")
        ax.legend()
    ax.set_ylabel("Initial nearest-node distance (mm)")
    fig.tight_layout()


def main():
    results = load_ssm_avoid_complex_results()
    plot_tracking_results(results)
    plot_motor_commands(results)
    plot_obstacle_view(results)
    plot_obstacle_distances(results)
    plt.show()


if __name__ == "__main__":
    main()
