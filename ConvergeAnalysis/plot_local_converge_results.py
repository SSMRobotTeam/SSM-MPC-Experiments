from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NUM_CASES = 6
DEFAULT_TIME_MIN = 0.0
DEFAULT_TIME_MAX = 2.0
DEFAULT_TIME_TICK = 0.5


def _load_struct(mat_path: Path, variable_name: str):
    mat_data = sciIO.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if variable_name not in mat_data:
        available = ", ".join(key for key in mat_data if not key.startswith("__"))
        raise KeyError(f"Missing variable '{variable_name}' in {mat_path}. Available: {available}")
    return mat_data[variable_name]


def _format_xyz_data(data_raw: np.ndarray, sample_count: int) -> np.ndarray:
    data = np.asarray(data_raw, dtype=float)
    if data.ndim != 2:
        if data.size == 3 * sample_count:
            return data.reshape(-1, 3)
        raise ValueError(f"Cannot interpret data shape {data.shape} as xyz samples.")

    rows, cols = data.shape
    if rows == sample_count and cols >= 3:
        return data[:, :3]
    if cols == sample_count and rows >= 3:
        return data[:3, :].T
    if data.size == 3 * sample_count:
        return data.reshape(-1, 3)
    raise ValueError(f"Cannot interpret data shape {data.shape} for {sample_count} samples.")


def load_case(case_index: int, data_dir: Path = SCRIPT_DIR) -> dict:
    comsol_name = f"comsol_save{case_index}"
    ssm_name = f"SSM_save{case_index}"
    comsol = _load_struct(data_dir / f"{comsol_name}.mat", comsol_name)
    ssm = _load_struct(data_dir / f"{ssm_name}.mat", ssm_name)

    t_comsol = np.asarray(comsol.time, dtype=float).reshape(-1)
    t_ssm = np.asarray(ssm.time, dtype=float).reshape(-1)
    xyz_comsol = _format_xyz_data(comsol.data, t_comsol.size)
    xyz_ssm = _format_xyz_data(ssm.data, t_ssm.size)

    comsol_order = np.argsort(t_comsol)
    ssm_order = np.argsort(t_ssm)
    return {
        "case": case_index,
        "t_comsol": t_comsol[comsol_order],
        "xyz_comsol": xyz_comsol[comsol_order],
        "t_ssm": t_ssm[ssm_order],
        "xyz_ssm": xyz_ssm[ssm_order],
        "ssm_rmse_saved": float(np.asarray(getattr(ssm, "RMSE", np.nan)).reshape(-1)[0]),
    }


def compute_case_rmse(case_data: dict) -> dict:
    t_comsol = case_data["t_comsol"]
    t_ssm = case_data["t_ssm"]
    xyz_comsol = case_data["xyz_comsol"]
    xyz_ssm = case_data["xyz_ssm"]

    t_start = max(float(t_comsol[0]), float(t_ssm[0]))
    t_end = min(float(t_comsol[-1]), float(t_ssm[-1]))
    t_common = t_comsol[(t_comsol >= t_start) & (t_comsol <= t_end)]
    if t_common.size < 2:
        raise ValueError(f"Case {case_data['case']} has fewer than two common time samples.")

    y_comsol = np.interp(t_common, t_comsol, xyz_comsol[:, 1])
    z_comsol = np.interp(t_common, t_comsol, xyz_comsol[:, 2])
    y_ssm = np.interp(t_common, t_ssm, xyz_ssm[:, 1])
    z_ssm = np.interp(t_common, t_ssm, xyz_ssm[:, 2])

    error_y = y_comsol - y_ssm
    error_z = z_comsol - z_ssm
    return {
        "case": case_data["case"],
        "t_common": t_common,
        # "rmse_y": float(np.sqrt(np.mean(error_y**2))),
        # "rmse_z": float(np.sqrt(np.mean(error_z**2))),
        "rmse_yz": float(np.sqrt(np.mean(error_y**2 + error_z**2))),
        "initial_displacement_yz_mm": float(np.linalg.norm(xyz_comsol[0, 1:3])),
        "ssm_rmse_saved": case_data["ssm_rmse_saved"],
    }


def _apply_time_axis(ax, use_custom_xlim: bool, time_min: float, time_max: float, time_tick: float):
    if use_custom_xlim:
        ax.set_xlim(time_min, time_max)
        ax.set_xticks(np.arange(time_min, time_max + 0.5 * time_tick, time_tick))


def plot_case(
    case_data: dict,
    output_dir: Path,
    *,
    use_custom_xlim: bool,
    time_min: float,
    time_max: float,
    time_tick: float,
    show: bool,
) -> None:
    case_index = case_data["case"]
    t_comsol = case_data["t_comsol"]
    xyz_comsol = case_data["xyz_comsol"]
    t_ssm = case_data["t_ssm"]
    xyz_ssm = case_data["xyz_ssm"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    channels = ((1, "Y"), (2, "Z"))
    for ax, (axis_idx, label) in zip(axes, channels):
        ax.plot(t_comsol, xyz_comsol[:, axis_idx], "b-", linewidth=2.0, label="COMSOL")
        ax.plot(t_ssm, xyz_ssm[:, axis_idx], "r--", linewidth=2.0, label="SSM")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{label} [mm]")
        ax.set_title(f"Case {case_index} - {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        _apply_time_axis(ax, use_custom_xlim, time_min, time_max, time_tick)

    fig.savefig(output_dir / f"case_{case_index:02d}_yz_compare.png", dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rmse_summary(rmse_rows: list[dict], output_dir: Path, *, show: bool) -> None:
    cases = np.asarray([row["case"] for row in rmse_rows], dtype=int)
    rmse_yz = np.asarray([row["rmse_yz"] for row in rmse_rows], dtype=float)
    initial_displacement = np.asarray([row["initial_displacement_yz_mm"] for row in rmse_rows], dtype=float)
    tick_labels = [f"Case {case}\nd0={initial_value:.1f} mm" for case, initial_value in zip(cases, initial_displacement)]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.plot(cases, rmse_yz, "o-", linewidth=2.0, label="RMSE YZ")
    # for case, rmse_value in zip(cases, rmse_yz):
    #     ax.annotate(
    #         f"RMSE={rmse_value:.2f} mm",
    #         xy=(case, rmse_value),
    #         xytext=(0, 9),
    #         textcoords="offset points",
    #         ha="center",
    #         va="bottom",
    #         fontsize=9,
    #     )
    # ax.set_xlabel("Case")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Local Convergence Analysis of the SSM Model")
    ax.set_xticks(cases)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "rmse_summary.png", dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    bars = ax.bar(cases, rmse_yz, width=0.5, label="YZ")
    for bar, rmse_value in zip(bars, rmse_yz):
        ax.annotate(
            f"RMSE={rmse_value:.2f} mm",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    # ax.set_xlabel("Case")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Local Convergence Analysis of the SSM Model")
    ax.set_xticks(cases)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "rmse_summary_bar.png", dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_rmse_csv(rmse_rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("case", "initial_displacement_yz_mm", "rmse_yz", "ssm_rmse_saved"),
        )
        writer.writeheader()
        for row in rmse_rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "initial_displacement_yz_mm": row["initial_displacement_yz_mm"],
                    "rmse_yz": row["rmse_yz"],
                    "ssm_rmse_saved": row["ssm_rmse_saved"],
                }
            )


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rmse_rows = []
    for case_index in range(1, int(args.num_cases) + 1):
        case_data = load_case(case_index, data_dir)
        rmse = compute_case_rmse(case_data)
        rmse_rows.append(rmse)
        plot_case(
            case_data,
            output_dir,
            use_custom_xlim=not args.auto_xlim,
            time_min=float(args.time_min),
            time_max=float(args.time_max),
            time_tick=float(args.time_tick),
            show=bool(args.show),
        )
        print(
            f"Case {case_index}: "
            f"RMSE_yz={rmse['rmse_yz']:.6f}"
        )

    plot_rmse_summary(rmse_rows, output_dir, show=bool(args.show))
    write_rmse_csv(rmse_rows, output_dir / "rmse_summary.csv")
    print(f"Saved figures and RMSE CSV to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local convergence comparison results.")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR, help="Folder containing COMSOL/SSM .mat files.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "figures", help="Folder for output figures.")
    parser.add_argument("--num-cases", type=int, default=DEFAULT_NUM_CASES, help="Number of cases to plot.")
    parser.add_argument("--time-min", type=float, default=DEFAULT_TIME_MIN, help="Custom plot time minimum.")
    parser.add_argument("--time-max", type=float, default=DEFAULT_TIME_MAX, help="Custom plot time maximum.")
    parser.add_argument("--time-tick", type=float, default=DEFAULT_TIME_TICK, help="Custom x-axis tick interval.")
    parser.add_argument("--auto-xlim", action="store_true", help="Use each case's natural time span.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
