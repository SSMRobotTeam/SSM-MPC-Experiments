from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import Any_plot_ssm_avoid_results as any_plot


DEFAULT_SAVE_PATH = any_plot.FIGURE_OUTPUT_DIR / any_plot.INDIVIDUAL_FIGURE_DIR_NAME / "d_obstacle_view.png"
OBSTACLE_VIEW_FIGSIZE = (8.2, 7.2)
SAVE_DPI = 400
OBSTACLE_Y_OFFSET_M = 0.010
LABEL_FONTSIZE = any_plot.COMBINED_LABEL_FONTSIZE + 6
TICK_FONTSIZE = any_plot.COMBINED_TICK_FONTSIZE + 4
LEGEND_FONTSIZE = any_plot.COMBINED_LEGEND_FONTSIZE + 6
TICK_NBINS = 3
GRID_COLOR = "0.65"
GRID_LINEWIDTH = 1.1


def set_3d_white_background(fig, ax):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = GRID_LINEWIDTH


def plot_d_obstacle_view():
    results = any_plot.load_ssm_avoid_complex_results()
    results = dict(results)
    results["pos_obstacle"] = results["pos_obstacle"].copy()
    if results["pos_obstacle"].size:
        results["pos_obstacle"][:, 1] += OBSTACLE_Y_OFFSET_M

    fig = plt.figure("SSM complex avoid anypoint obstacle view", figsize=OBSTACLE_VIEW_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    any_plot._plot_obstacle_view_on_axis(results, ax)
    set_3d_white_background(fig, ax)
    ax.grid(True)
    ax.view_init(elev=36, azim=-61, roll=2)

    ax.set_xlabel("X (mm)", fontsize=LABEL_FONTSIZE, labelpad=12)
    ax.set_ylabel("Y (mm)", fontsize=LABEL_FONTSIZE, labelpad=12)
    ax.set_zlabel("Z (mm)", fontsize=LABEL_FONTSIZE, labelpad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_NBINS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_NBINS))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=TICK_NBINS))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, pad=2)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, pad=2)
    ax.tick_params(axis="z", labelsize=TICK_FONTSIZE, pad=2)

    legend = ax.get_legend()
    if legend is not None:
        legend.set_loc("upper left")
        legend.set_in_layout(False)
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_FONTSIZE)

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.96)
    return fig


def save_d_obstacle_view(fig, output_path: str | Path = DEFAULT_SAVE_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, facecolor="white")
    return output_path


def close_and_save_on_s(fig, output_path: str | Path = DEFAULT_SAVE_PATH):
    def _on_key_press(event):
        if event.key not in ("s", "S"):
            return

        saved_path = save_d_obstacle_view(fig, output_path)
        print(f"Saved d_obstacle_view figure to: {saved_path}")
        plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key_press)


def parse_args():
    parser = ArgumentParser(
        description="Show only the d_obstacle_view panel for the 03anypoint SSM avoidance results.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_SAVE_PATH,
        help="Output image path used when pressing S. Defaults to figures/individual_subplots/d_obstacle_view.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fig = plot_d_obstacle_view()
    close_and_save_on_s(fig, args.output)
    print(f"Press S in the figure window to save at {SAVE_DPI} dpi and close.")
    plt.show()


if __name__ == "__main__":
    main()
