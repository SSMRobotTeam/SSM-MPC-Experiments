from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamics.SSM.matlab_data import ROM_DIR, load_rom_data
from dynamics.SSM.ssm_dynamics import auto_red_dyn


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


def _time_vector(
    est_data: dict[str, np.ndarray],
    raw_data: dict[str, np.ndarray] | None,
    preferred: str,
) -> np.ndarray:
    if preferred in est_data:
        return np.asarray(est_data[preferred], dtype=float)
    if raw_data is not None and preferred in raw_data:
        return np.asarray(raw_data[preferred], dtype=float)
    raise KeyError(f"Missing time column: {preferred}")


def _pk_matrix(est_data: dict[str, np.ndarray]) -> np.ndarray:
    pk_columns = sorted(
        (name for name in est_data if name.startswith("pk_")),
        key=lambda name: int(name.split("_", maxsplit=1)[1]),
    )
    if not pk_columns:
        raise KeyError("Estimate CSV is missing pk_* columns.")
    return np.column_stack([np.asarray(est_data[column], dtype=float) for column in pk_columns])


def estimate_s0_sequence(
    time_s: np.ndarray,
    pk_values: np.ndarray,
    au_data,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate S0(t) from pk(t) using S0 = dp/dt - R(p)."""
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    pk_values = np.asarray(pk_values, dtype=float)
    if pk_values.ndim != 2:
        raise ValueError(f"pk_values must be 2-D, got shape {pk_values.shape}.")
    if time_s.size != pk_values.shape[0]:
        raise ValueError(f"time_s has {time_s.size} samples, pk has {pk_values.shape[0]} rows.")
    if pk_values.shape[0] == 0:
        return pk_values.copy(), pk_values.copy(), pk_values.copy()
    if pk_values.shape[0] == 1:
        pk_dot = np.zeros_like(pk_values)
    else:
        finite_time = np.all(np.isfinite(time_s))
        strictly_increasing = np.all(np.diff(time_s) > 0.0)
        if not finite_time or not strictly_increasing:
            raise ValueError("Time values must be finite and strictly increasing to estimate dp/dt.")
        edge_order = 2 if pk_values.shape[0] >= 3 else 1
        pk_dot = np.gradient(pk_values, time_s, axis=0, edge_order=edge_order)

    r_values = auto_red_dyn(pk_values.T, au_data).T
    s0_values = pk_dot - r_values
    return pk_dot, r_values, s0_values


def write_s0_csv(
    output_path: Path,
    est_data: dict[str, np.ndarray],
    time_s: np.ndarray,
    pk_values: np.ndarray,
    pk_dot: np.ndarray,
    r_values: np.ndarray,
    s0_values: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_count = pk_values.shape[1]
    headers = ["row_index", "step_index", "time_s"]
    headers.extend([f"pk_{idx + 1}" for idx in range(state_count)])
    headers.extend([f"pk_dot_{idx + 1}" for idx in range(state_count)])
    headers.extend([f"R_pk_{idx + 1}" for idx in range(state_count)])
    headers.extend([f"S0_{idx + 1}" for idx in range(state_count)])

    row_index = est_data.get("row_index", np.arange(pk_values.shape[0], dtype=float))
    step_index = est_data.get("step_index", row_index)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row_idx in range(pk_values.shape[0]):
            row = {
                "row_index": int(row_index[row_idx]),
                "step_index": int(step_index[row_idx]),
                "time_s": float(time_s[row_idx]),
            }
            for idx in range(state_count):
                row[f"pk_{idx + 1}"] = float(pk_values[row_idx, idx])
                row[f"pk_dot_{idx + 1}"] = float(pk_dot[row_idx, idx])
                row[f"R_pk_{idx + 1}"] = float(r_values[row_idx, idx])
                row[f"S0_{idx + 1}"] = float(s0_values[row_idx, idx])
            writer.writerow(row)


def plot_s0_time_series(
    time_s: np.ndarray,
    s0_values: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_count = s0_values.shape[1]
    fig, axes = plt.subplots(state_count, 1, figsize=(11, max(8, 1.7 * state_count)), sharex=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    color_cycle = plt.get_cmap("tab10")
    for idx, ax in enumerate(axes):
        ax.plot(time_s, s0_values[:, idx], color=color_cycle(idx % 10), linewidth=1.5)
        ax.set_ylabel(f"S0_{idx + 1}")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    axes[0].set_title(r"Estimated $S_0(t)=\dot{p}-R(p)$")
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(output_path, dpi=300)
    return fig


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results" / "S0_est"
    parser = argparse.ArgumentParser(
        description="Estimate S0(t)=dp/dt-R(p) from DynamicVerification pk estimates.",
    )
    parser.add_argument(
        "--estimate",
        type=Path,
        default=script_dir / "results" / "dynamic_pk_estimates.csv",
        help="CSV generated by estimate_dynamic_pk.py.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "DynData" / "Dyndata01.csv",
        help="Optional raw dynamic CSV used only if the selected time column is absent from --estimate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_dir / "estimated_S0_sequence.csv",
        help="Output CSV for pk, dp/dt, R(pk), and S0.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=output_dir / "estimated_S0_sequence.png",
        help="Output figure for the estimated S0 time series.",
    )
    parser.add_argument(
        "--time-column",
        default="actual_time_s",
        choices=("actual_time_s", "planned_time_s"),
        help="Time column used for dp/dt.",
    )
    parser.add_argument("--show", action="store_true", help="Show the S0 figure interactively after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    est_data = _load_csv(args.estimate)
    raw_data = _load_csv(args.data) if args.data.exists() else None
    time_s = _time_vector(est_data, raw_data, args.time_column)
    pk_values = _pk_matrix(est_data)

    rom_data = load_rom_data(ROM_DIR)
    pk_dot, r_values, s0_values = estimate_s0_sequence(time_s, pk_values, rom_data["auData"])

    write_s0_csv(args.output, est_data, time_s, pk_values, pk_dot, r_values, s0_values)
    fig = plot_s0_time_series(time_s, s0_values, args.plot)

    print(f"Estimated S0 for {pk_values.shape[0]} samples with {pk_values.shape[1]} states.")
    print(f"Saved S0 CSV to {args.output}")
    print(f"Saved S0 plot to {args.plot}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
