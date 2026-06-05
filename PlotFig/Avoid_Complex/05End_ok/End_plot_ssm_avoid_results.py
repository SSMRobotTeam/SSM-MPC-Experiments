from pathlib import Path

from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy.io as sciIO


PLOT_DIR = Path(__file__).resolve().parent

SUFFIX = "_complex"
AVOID_RESULTS_MAT = f"ssm_avoid_python_results{SUFFIX}.mat"
END_INITIAL_POSITION_M = np.array([0.0, 0.0, 0.4400])
DEFAULT_OBSTACLE_RADIUS_M = 0.02
TRAJECTORY_ANIMATION_DURATION_S = 15.0
TRAJECTORY_ANIMATION_FPS = 10
TRAJECTORY_ANIMATION_NAME = "ssm_measured_end_trajectory_animation.mp4"
TRAJECTORY_ANIMATION_FIGSIZE = (16, 9)
TRAJECTORY_ANIMATION_DPI = 160
TRAJECTORY_ANIMATION_LABEL_FONTSIZE = 22
TRAJECTORY_ANIMATION_TICK_FONTSIZE = 18
TRAJECTORY_ANIMATION_LEGEND_FONTSIZE = 18
FIGURE_OUTPUT_DIR = PLOT_DIR / "figures"
COMBINED_FIGURE_NAME = "ssm_complex_avoid_end_results.png"
COMBINED_FIGURE_DPI = 300
COMBINED_FIGURE_FIGSIZE = (22, 10.5)
COMBINED_TITLE_FONTSIZE = 18
COMBINED_LABEL_FONTSIZE = 15
COMBINED_TICK_FONTSIZE = 13
COMBINED_LEGEND_FONTSIZE = 14
TOP_RIGHT_IMAGE_NAME = "EndAvoid.png"
BOTTOM_RIGHT_IMAGE_NAME = "RealPics.jpg"
COMBINED_PANEL_LABEL_FONTSIZE = 17
INDIVIDUAL_FIGURE_DIR_NAME = "individual_subplots"
INDIVIDUAL_FIGURE_DPI = 300


def load_ssm_avoid_complex_results(plot_dir: str | Path = PLOT_DIR):
    plot_dir = Path(plot_dir)
    mat_path = plot_dir / AVOID_RESULTS_MAT
    results = {
        "t": np.load(plot_dir / f"t_ssm_avoid{SUFFIX}.npy"),
        "y_exprm": np.load(plot_dir / f"y_exprm_ssm_avoid{SUFFIX}.npy"),
        "y_ssm": None,
        "ut_save": np.load(plot_dir / f"ut_save_ssm_avoid{SUFFIX}.npy"),
        "zd": None,
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

    if mat_path.exists():
        mat_data = sciIO.loadmat(str(mat_path), squeeze_me=True)
        if "t" in mat_data:
            results["t"] = np.asarray(mat_data["t"], dtype=float).reshape(-1)
        if "y_exprm" in mat_data:
            results["y_exprm"] = np.asarray(mat_data["y_exprm"], dtype=float)
        if "ut_save" in mat_data:
            results["ut_save"] = np.asarray(mat_data["ut_save"], dtype=float)
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


def _style_axis_fonts(ax):
    ax.title.set_fontsize(COMBINED_TITLE_FONTSIZE)
    ax.xaxis.label.set_size(COMBINED_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(COMBINED_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=COMBINED_TICK_FONTSIZE)
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_size(COMBINED_LABEL_FONTSIZE)
        ax.tick_params(axis="z", labelsize=COMBINED_TICK_FONTSIZE)


def _add_image_on_axis(ax, image_path: Path):
    ax.set_axis_off()
    if not image_path.exists():
        ax.text(
            0.5,
            0.5,
            f"Missing image\n{image_path.name}",
            ha="center",
            va="center",
            fontsize=COMBINED_LABEL_FONTSIZE,
        )
        return ax

    image = plt.imread(image_path)
    ax.imshow(image)
    ax.set_anchor("W")
    return ax


def _trim_png_whitespace(path: Path, margin_px: int = 28):
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


def _adjust_axis_box(ax, *, dx=0.0, dy=0.0, width_scale=1.0, height_scale=1.0, anchor="center"):
    box = ax.get_position()
    if anchor == "left":
        x0 = box.x0 + dx
        x1 = x0 + box.width * width_scale
    elif anchor == "right":
        x1 = box.x1 + dx
        x0 = x1 - box.width * width_scale
    else:
        center_x = 0.5 * (box.x0 + box.x1) + dx
        new_width = box.width * width_scale
        x0 = center_x - 0.5 * new_width
        x1 = center_x + 0.5 * new_width

    if anchor in ("bottom",):
        y0 = box.y0 + dy
        y1 = y0 + box.height * height_scale
    else:
        center_y = 0.5 * (box.y0 + box.y1) + dy
        new_height = box.height * height_scale
        y0 = center_y - 0.5 * new_height
        y1 = center_y + 0.5 * new_height

    ax.set_position([x0, y0, x1 - x0, y1 - y0])


def _axes_group_bbox(axes):
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    return x0, y0, x1, y1


def _add_panel_label(fig, axes, label: str):
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    x0, _, _, y1 = _axes_group_bbox(axes)
    fig.text(
        x0,
        y1 + 0.01,
        label,
        ha="left",
        va="bottom",
        fontsize=COMBINED_PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )


def _plot_tracking_on_axes(results, axes, show_legend=True, xlabel_all=False):
    t = results["t"]
    y = results["y_exprm"]
    zd = results.get("zd")
    n = min(t.size, y.shape[0])

    axis_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    legend_handles = []
    legend_labels = []
    for idx, (ax, ylabel) in enumerate(zip(axes, axis_labels)):
        measured_line, = ax.plot(t[:n], y[:n, idx], label="measured")
        if zd is not None:
            ref_n = min(n, zd.shape[1])
            desired_line, = ax.plot(t[:ref_n], zd[idx, :ref_n] * 1000.0, "k--", label="desired")
            if idx == 0:
                legend_handles.extend([desired_line, measured_line])
                legend_labels.extend(["desired", "measured"])
        elif idx == 0:
            legend_handles.append(measured_line)
            legend_labels.append("measured")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if xlabel_all:
            ax.set_xlabel("Time (s)")
        _style_axis_fonts(ax)
    if not xlabel_all:
        axes[-1].set_xlabel("Time (s)")
        _style_axis_fonts(axes[-1])
    if show_legend and legend_handles:
        axes[0].legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=COMBINED_LEGEND_FONTSIZE,
        )
    return legend_handles, legend_labels


def _plot_tracking_component_on_axis(results, ax, component_idx: int, show_legend=True):
    t = results["t"]
    y = results["y_exprm"]
    zd = results.get("zd")
    n = min(t.size, y.shape[0])
    axis_labels = ["X (mm)", "Y (mm)", "Z (mm)"]

    measured_line, = ax.plot(t[:n], y[:n, component_idx], label="measured")
    handles = [measured_line]
    labels = ["measured"]
    if zd is not None:
        ref_n = min(n, zd.shape[1])
        desired_line, = ax.plot(t[:ref_n], zd[component_idx, :ref_n] * 1000.0, "k--", label="desired")
        handles = [desired_line, measured_line]
        labels = ["desired", "measured"]

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(axis_labels[component_idx])
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=COMBINED_LEGEND_FONTSIZE)
    _style_axis_fonts(ax)
    return ax


def plot_tracking_results(results):
    # fig = plt.figure("avoid and tracking")
    fig = plt.figure()
    axes = [fig.add_subplot(3, 1, idx + 1) for idx in range(3)]
    _plot_tracking_on_axes(results, axes)
    fig.tight_layout()
    return fig


def plot_tracking_group_panel(results):
    fig = plt.figure("SSM complex avoid tracking panel", figsize=(14, 4.0), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, idx + 1) for idx in range(3)]
    handles, labels = _plot_tracking_on_axes(results, axes, show_legend=False, xlabel_all=True)
    if handles:
        by_label = dict(zip(labels, handles))
        axes[0].legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            ncol=len(by_label),
            frameon=False,
            fontsize=COMBINED_LEGEND_FONTSIZE,
        )
    return fig


def plot_tracking_component(results, component_idx: int):
    axis_names = ["X", "Y", "Z"]
    fig = plt.figure(f"SSM complex avoid {axis_names[component_idx]} tracking", figsize=(6.2, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    _plot_tracking_component_on_axis(results, ax, component_idx)
    fig.tight_layout()
    return fig


def _plot_motor_commands_on_axis(results, ax):
    t, u = _trim_time_series(results["t"], results["ut_save"])

    styles = ["r-", "g-", "b-", "r--", "g--", "b--"]
    for idx in range(u.shape[1]):
        ax.plot(t, u[:, idx], styles[idx % len(styles)], label=f"u{idx + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor command (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=COMBINED_LEGEND_FONTSIZE)
    _style_axis_fonts(ax)
    return ax


def plot_motor_commands(results):
    # fig = plt.figure("SSM complex avoid motor commands")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    _plot_motor_commands_on_axis(results, ax)
    fig.tight_layout()
    return fig


def _plot_obstacle_view_on_axis(results, ax, legend_outside=False):
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

    ax.plot(
        measured_end_mm[:, 0],
        measured_end_mm[:, 1],
        measured_end_mm[:, 2],
        color="tab:blue",
        label="measured",
    )

    # if y_ssm is not None:
    #     count = min(t.size, y_ssm.shape[0])
    #     algorithm_end_mm = y_ssm[:count, :3] + pos_end_mm
    #     ax.plot(
    #         algorithm_end_mm[:, 0],
    #         algorithm_end_mm[:, 1],
    #         algorithm_end_mm[:, 2],
    #         color="tab:orange",
    #         label="SSM calculated end",
    #     )

    if zd is not None:
        ref_n = min(t.size, zd.shape[1])
        desired_end_mm = zd[:3, :ref_n].T * 1000.0 + pos_end_mm
        ax.plot(
            desired_end_mm[:, 0],
            desired_end_mm[:, 1],
            desired_end_mm[:, 2],
            "k--",
            label="desired",
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

        # if obstacle_idx < node_positions_m.shape[0]:
        #     node_mm = node_positions_m[obstacle_idx] * 1000.0
        #     node_label = f"nearest node {node_indices[obstacle_idx]}" if obstacle_idx < node_indices.size else "nearest node"
        #     ax.scatter(
        #         node_mm[0],
        #         node_mm[1],
        #         node_mm[2],
        #         c="tab:green",
        #         marker="o",
        #         s=45,
        #         label=node_label,
        #     )
        #     ax.plot(
        #         [obstacle_mm[0], node_mm[0]],
        #         [obstacle_mm[1], node_mm[1]],
        #         [obstacle_mm[2], node_mm[2]],
        #         color="tab:green",
        #         linestyle=":",
        #         linewidth=0.8,
        #     )

    labelpad = 10 if legend_outside else None
    ax.set_xlabel("X (mm)", labelpad=labelpad)
    ax.set_ylabel("Y (mm)", labelpad=labelpad)
    ax.set_zlabel("Z (mm)", labelpad=labelpad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    _set_axes_equal_3d(ax)
    z_lower, z_upper = ax.get_zlim3d()
    ax.set_zlim3d(z_upper, z_lower)
    ax.view_init(elev=27, azim=-69, roll=0)
    _style_axis_fonts(ax)
    if legend_outside:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.9),
            borderaxespad=0.0,
            ncol=3,
            frameon=False,
            fontsize=COMBINED_LEGEND_FONTSIZE - 2,
            handlelength=1.8,
            columnspacing=1.2,
        )
        ax.xaxis.label.set_size(COMBINED_LABEL_FONTSIZE - 2)
        ax.yaxis.label.set_size(COMBINED_LABEL_FONTSIZE - 2)
        ax.zaxis.label.set_size(COMBINED_LABEL_FONTSIZE - 2)
        ax.tick_params(axis="both", labelsize=COMBINED_TICK_FONTSIZE - 2)
        ax.tick_params(axis="z", labelsize=COMBINED_TICK_FONTSIZE - 2)
    else:
        ax.legend(fontsize=COMBINED_LEGEND_FONTSIZE)
    return ax


def _set_3d_white_background(fig, ax):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))


def plot_obstacle_view(results):
    # fig = plt.figure("SSM complex avoid obstacle view")
    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _plot_obstacle_view_on_axis(results, ax, legend_outside=True)
    _set_3d_white_background(fig, ax)
    ax.grid(False)
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.18, top=0.78)
    return fig


def save_combined_results_figure(fig, output_dir: str | Path = FIGURE_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / COMBINED_FIGURE_NAME
    fig.savefig(output_path, dpi=COMBINED_FIGURE_DPI, bbox_inches="tight")
    return output_path


def plot_image_panel(image_path: Path, figure_name: str):
    image_path = Path(image_path)
    if image_path.exists():
        image = plt.imread(image_path)
        height, width = image.shape[:2]
        aspect = max(width / max(height, 1), 0.25)
        fig_width = min(max(4.0 * aspect, 3.0), 8.0)
        fig_height = min(max(4.0 / aspect, 3.0), 8.0)
    else:
        fig_width, fig_height = 5.0, 4.0

    fig = plt.figure(figure_name, figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1)
    _add_image_on_axis(ax, image_path)
    fig.tight_layout(pad=0.02)
    return fig


def _save_and_close_figure(
    fig,
    output_path: Path,
    dpi: int = INDIVIDUAL_FIGURE_DPI,
    bbox_inches="tight",
    trim_whitespace=False,
    facecolor=None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
    plt.close(fig)
    if trim_whitespace:
        _trim_png_whitespace(output_path)
    return output_path


def save_individual_subplots(results, output_dir: str | Path = FIGURE_OUTPUT_DIR):
    output_dir = Path(output_dir) / INDIVIDUAL_FIGURE_DIR_NAME
    saved_paths = []

    saved_paths.append(_save_and_close_figure(plot_tracking_group_panel(results), output_dir / "a_tracking_xyz_panel.png"))
    for component_idx, axis_name in enumerate(("x", "y", "z")):
        saved_paths.append(
            _save_and_close_figure(
                plot_tracking_component(results, component_idx),
                output_dir / f"a_{axis_name}_tracking.png",
            )
        )

    saved_paths.append(
        _save_and_close_figure(
            plot_image_panel(FIGURE_OUTPUT_DIR / TOP_RIGHT_IMAGE_NAME, "EndAvoid panel"),
            output_dir / "b_end_avoid.png",
        )
    )
    saved_paths.append(_save_and_close_figure(plot_motor_commands(results), output_dir / "c_motor_commands.png"))
    saved_paths.append(
        _save_and_close_figure(
            plot_obstacle_view(results),
            output_dir / "d_obstacle_view.png",
            bbox_inches=None,
            trim_whitespace=True,
            facecolor="white",
        )
    )
    saved_paths.append(
        _save_and_close_figure(
            plot_image_panel(FIGURE_OUTPUT_DIR / BOTTOM_RIGHT_IMAGE_NAME, "RealPics panel"),
            output_dir / "e_real_pics.png",
        )
    )
    return saved_paths


def plot_combined_results(results):
    fig = plt.figure("SSM complex avoid end results", figsize=COMBINED_FIGURE_FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.035, wspace=0.008, hspace=0.08)
    grid = fig.add_gridspec(2, 8, height_ratios=(1.0, 1.25), width_ratios=(1, 1, 1, 1, 1, 1, 1.18, 1.18))

    tracking_axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
    ]
    top_image_ax = fig.add_subplot(grid[0, 6:8])
    motor_ax = fig.add_subplot(grid[1, 0:3])
    obstacle_ax = fig.add_subplot(grid[1, 3:6], projection="3d")
    bottom_image_ax = fig.add_subplot(grid[1, 6:8])

    tracking_handles, tracking_labels = _plot_tracking_on_axes(
        results,
        tracking_axes,
        show_legend=False,
        xlabel_all=True,
    )
    _plot_motor_commands_on_axis(results, motor_ax)
    _plot_obstacle_view_on_axis(results, obstacle_ax)
    _add_image_on_axis(top_image_ax, FIGURE_OUTPUT_DIR / TOP_RIGHT_IMAGE_NAME)
    _add_image_on_axis(bottom_image_ax, FIGURE_OUTPUT_DIR / BOTTOM_RIGHT_IMAGE_NAME)
    obstacle_ax.set_anchor("W")

    # tracking_axes[0].set_title("X tracking", fontsize=COMBINED_TITLE_FONTSIZE)
    # tracking_axes[1].set_title("Y tracking", fontsize=COMBINED_TITLE_FONTSIZE)
    # tracking_axes[2].set_title("Z tracking", fontsize=COMBINED_TITLE_FONTSIZE)
    # obstacle_ax.set_title("SSM complex avoid obstacle view", fontsize=COMBINED_TITLE_FONTSIZE)
    # motor_ax.set_title("SSM complex avoid motor commands", fontsize=COMBINED_TITLE_FONTSIZE)

    if tracking_handles:
        by_label = dict(zip(tracking_labels, tracking_handles))
        tracking_axes[0].legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            ncol=len(by_label),
            frameon=False,
            fontsize=COMBINED_LEGEND_FONTSIZE,
        )

    fig.canvas.draw()
    fig.set_constrained_layout(False)
    _adjust_axis_box(top_image_ax, dx=-0.006, width_scale=1.08, height_scale=1.2, anchor="left")
    _adjust_axis_box(obstacle_ax, dx=-0.018, width_scale=1.03, height_scale=1.04, anchor="center")
    _adjust_axis_box(bottom_image_ax, dx=-0.024, width_scale=1.16, height_scale=1.14, anchor="left")
    fig.canvas.draw()
    _add_panel_label(fig, tracking_axes, "(a)")
    _add_panel_label(fig, top_image_ax, "(b)")
    _add_panel_label(fig, motor_ax, "(c)")
    _add_panel_label(fig, obstacle_ax, "(d)")
    _add_panel_label(fig, bottom_image_ax, "(e)")

    return fig


def plot_obstacle_view_animation(results):
    t, y = _trim_time_series(results["t"], results["y_exprm"])
    zd = results.get("zd")
    pos_end_mm = np.asarray(results.get("pos_end", END_INITIAL_POSITION_M), dtype=float).reshape(3) * 1000.0
    obstacles_m = np.asarray(results.get("pos_obstacle", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    obstacle_radius_mm = float(results.get("obstacle_radius", DEFAULT_OBSTACLE_RADIUS_M)) * 1000.0
    d_safe = results.get("d_safe")
    measured_end_mm = y[:, :3] + pos_end_mm
    if measured_end_mm.size == 0:
        return None

    frame_count = max(2, int(TRAJECTORY_ANIMATION_DURATION_S * TRAJECTORY_ANIMATION_FPS))
    frame_indices = np.unique(np.linspace(0, measured_end_mm.shape[0] - 1, frame_count).astype(int))
    interval_ms = 1000.0 * TRAJECTORY_ANIMATION_DURATION_S / frame_indices.size

    fig = plt.figure("SSM complex avoid measured end animation", figsize=TRAJECTORY_ANIMATION_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(
        measured_end_mm[:, 0],
        measured_end_mm[:, 1],
        measured_end_mm[:, 2],
        color="tab:blue",
        alpha=0.35,
        label="measured",
    ) 
    if zd is not None:
        ref_n = min(t.size, zd.shape[1])
        desired_end_mm = zd[:3, :ref_n].T * 1000.0 + pos_end_mm
        ax.plot(
            desired_end_mm[:, 0],
            desired_end_mm[:, 1],
            desired_end_mm[:, 2],
            "k--",
            label="desired",
        )

    trail_line, = ax.plot([], [], [], color="tab:blue", linewidth=2.0, label="animated path")
    current_point = ax.scatter(
        measured_end_mm[0, 0],
        measured_end_mm[0, 1],
        measured_end_mm[0, 2],
        c="k",
        s=55,
        label="current end",
        depthshade=False,
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

    ax.set_xlabel("X (mm)", fontsize=TRAJECTORY_ANIMATION_LABEL_FONTSIZE, labelpad=16)
    ax.set_ylabel("Y (mm)", fontsize=TRAJECTORY_ANIMATION_LABEL_FONTSIZE, labelpad=16)
    ax.set_zlabel("Z (mm)", fontsize=TRAJECTORY_ANIMATION_LABEL_FONTSIZE, labelpad=16)
    ax.tick_params(axis="both", labelsize=TRAJECTORY_ANIMATION_TICK_FONTSIZE, pad=8)
    ax.tick_params(axis="z", labelsize=TRAJECTORY_ANIMATION_TICK_FONTSIZE, pad=8)
    _set_axes_equal_3d(ax)
    z_lower, z_upper = ax.get_zlim3d()
    ax.set_zlim3d(z_upper, z_lower)
    ax.view_init(elev=27, azim=-69, roll=0)
    ax.legend(fontsize=TRAJECTORY_ANIMATION_LEGEND_FONTSIZE)
    fig.tight_layout()

    def update(frame_idx):
        sample_idx = frame_indices[frame_idx]
        point = measured_end_mm[sample_idx]
        trail = measured_end_mm[: sample_idx + 1]
        trail_line.set_data(trail[:, 0], trail[:, 1])
        trail_line.set_3d_properties(trail[:, 2])
        current_point._offsets3d = ([point[0]], [point[1]], [point[2]])
        # ax.set_title(f"Measured end trajectory playback: {TRAJECTORY_ANIMATION_DURATION_S:.0f} s")
        return trail_line, current_point

    return FuncAnimation(
        fig,
        update,
        frames=frame_indices.size,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )


def save_trajectory_animation(animation, plot_dir: str | Path = PLOT_DIR):
    if animation is None:
        return None

    output_path = Path(plot_dir) / TRAJECTORY_ANIMATION_NAME
    writer = FFMpegWriter(fps=TRAJECTORY_ANIMATION_FPS, bitrate=1800)
    animation.save(output_path, writer=writer, dpi=TRAJECTORY_ANIMATION_DPI)
    return output_path


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
    # combined_fig = plot_combined_results(results)
    # output_path = save_combined_results_figure(combined_fig)
    # print(f"Saved combined results figure to: {output_path}")
    individual_paths = save_individual_subplots(results)
    print(f"Saved {len(individual_paths)} individual subplot figure(s) to: {FIGURE_OUTPUT_DIR / INDIVIDUAL_FIGURE_DIR_NAME}")
    # trajectory_animation = plot_obstacle_view_animation(results)
    # save_trajectory_animation(trajectory_animation)
    # plot_obstacle_distances(results)
    plt.show()
    # return trajectory_animation


if __name__ == "__main__":
    main()
