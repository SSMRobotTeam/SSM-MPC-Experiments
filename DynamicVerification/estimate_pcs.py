from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Main_demo_pcs import END_OFFSET, make_robot, zero_pcs_strains
from zmq_message.zmq_control import reorder_motor_positions


AXES = ("x", "y", "z")
RAW_U_COLUMNS = tuple(f"raw_u{idx}_mm" for idx in range(1, 7))
MOTOR_COLUMNS = tuple(f"motor_sent_u{idx}_mm" for idx in range(1, 7))
MOTOR_COUNT = 6

# Least-squares residual weights. Tune these defaults here, or override them with CLI arguments.
# DEFAULT_CABLE_WEIGHT = 2.5
# DEFAULT_ENDEFF_WEIGHT = 0.1
# DEFAULT_REGULARIZATION_WEIGHT = 0.001

DEFAULT_CABLE_WEIGHT = 2.5
DEFAULT_ENDEFF_WEIGHT = 0
DEFAULT_REGULARIZATION_WEIGHT = 0.001


def swap_cable_pairs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(MOTOR_COUNT)
    # return values[[0, 2, 1, 3, 5, 4]]
    return values[[0, 1, 2, 3, 4, 5]]


def _load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _raw_u_from_row(row: dict[str, object]) -> np.ndarray:
    return np.asarray([float(row[column]) for column in RAW_U_COLUMNS], dtype=float)


def _motor_sent_from_row(row: dict[str, object]) -> np.ndarray:
    return np.asarray([float(row[column]) for column in MOTOR_COLUMNS], dtype=float)


def _endeff_measurement_mm(row: dict[str, object]) -> np.ndarray:
    values = np.asarray([float(row[f"endeff_disp_{axis}_ref_mm"]) for axis in AXES], dtype=float)
    values[0] *= -1.0
    return values


def _pcs_residual(
    q_flat: np.ndarray,
    robot,
    q_guess: np.ndarray,
    baseline_lengths_m: np.ndarray,
    cable_target_mm: np.ndarray,
    endeff_target_mm: np.ndarray,
    *,
    cable_weight: float,
    endeff_weight: float,
    regularization_weight: float,
) -> np.ndarray:
    q = np.asarray(q_flat, dtype=float).reshape(q_guess.shape)
    cable_delta_m = robot.tendon_lengths(q) - baseline_lengths_m
    cable_residual = np.sqrt(cable_weight) * (cable_delta_m - cable_target_mm.reshape(6) / 1000.0)
    endeff_target_m = np.asarray(END_OFFSET, dtype=float).reshape(3) + endeff_target_mm.reshape(3) / 1000.0
    endeff_residual = np.sqrt(endeff_weight) * (robot.tip_position(q) - endeff_target_m)
    regularization_residual = np.sqrt(regularization_weight) * (q.reshape(-1) - q_guess.reshape(-1))
    return np.concatenate([cable_residual, endeff_residual, regularization_residual])


def estimate_q(
    robot,
    q_guess: np.ndarray,
    baseline_lengths_m: np.ndarray,
    cable_target_mm: np.ndarray,
    endeff_target_mm: np.ndarray,
    *,
    max_nfev: int,
    cable_weight: float,
    endeff_weight: float,
    regularization_weight: float,
) -> tuple[np.ndarray, bool, float, float, float]:
    start_time = time.perf_counter()
    result = least_squares(
        lambda q_flat: _pcs_residual(
            q_flat,
            robot,
            q_guess,
            baseline_lengths_m,
            cable_target_mm,
            endeff_target_mm,
            cable_weight=cable_weight,
            endeff_weight=endeff_weight,
            regularization_weight=regularization_weight,
        ),
        q_guess.reshape(-1),
        method="lm",
        max_nfev=max_nfev,
    )
    elapsed = time.perf_counter() - start_time
    residual = _pcs_residual(
        result.x,
        robot,
        q_guess,
        baseline_lengths_m,
        cable_target_mm,
        endeff_target_mm,
        cable_weight=cable_weight,
        endeff_weight=endeff_weight,
        regularization_weight=regularization_weight,
    )
    cable_residual_norm = float(np.linalg.norm(residual[:MOTOR_COUNT], 2))
    endeff_residual_norm = float(np.linalg.norm(residual[MOTOR_COUNT : MOTOR_COUNT + 3], 2))
    return result.x.reshape(q_guess.shape), cable_residual_norm < 1e-2, cable_residual_norm, endeff_residual_norm, elapsed


def _integration_time_s(results: list[dict[str, object]]) -> np.ndarray:
    values = []
    for idx, result in enumerate(results):
        try:
            values.append(float(result.get("actual_time_s", idx)))
        except (TypeError, ValueError):
            values.append(float(idx))
    time_s = np.asarray(values, dtype=float)
    if time_s.size > 1 and np.all(np.diff(time_s) >= 0):
        return time_s
    return np.arange(len(results), dtype=float)


def _integral_squared_error(time_s: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, float]:
    squared = np.asarray(errors, dtype=float) ** 2
    if time_s.size > 1:
        per_axis = np.trapezoid(squared, x=time_s, axis=0)
    else:
        per_axis = np.zeros(squared.shape[1], dtype=float)
    return per_axis, float(np.sum(per_axis))


def _format_vector(values: np.ndarray, precision: int = 8) -> str:
    return "[" + ", ".join(f"{value:.{precision}g}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def estimate_rows(
    rows: list[dict[str, object]],
    *,
    sequential: bool,
    max_nfev: int,
    cable_weight: float,
    endeff_weight: float,
    regularization_weight: float,
) -> list[dict[str, object]]:
    robot = make_robot()
    q_zero = zero_pcs_strains(robot)
    baseline_lengths_m = robot.tendon_lengths(q_zero)
    q_guess = q_zero.copy()
    results = []

    for row_idx, row in enumerate(rows):
        if not sequential:
            q_guess = q_zero.copy()
        raw_u_mm = _raw_u_from_row(row)
        cable_target_mm = swap_cable_pairs(raw_u_mm)
        endeff_target_mm = _endeff_measurement_mm(row)
        q_est, accepted, cable_residual_norm, endeff_residual_norm, elapsed = estimate_q(
            robot,
            q_guess,
            baseline_lengths_m,
            cable_target_mm,
            endeff_target_mm,
            max_nfev=max_nfev,
            cable_weight=cable_weight,
            endeff_weight=endeff_weight,
            regularization_weight=regularization_weight,
        )
        cable_length_m = robot.tendon_lengths(q_est)
        cable_delta_mm = (cable_length_m - baseline_lengths_m) * 1000.0
        pcs_endeff_position_mm = robot.tip_position(q_est) * 1000.0
        pcs_endeff_displacement_mm = pcs_endeff_position_mm - np.asarray(END_OFFSET, dtype=float) * 1000.0
        motor_order_delta_mm = np.asarray(reorder_motor_positions(cable_delta_mm), dtype=float)

        results.append(
            {
                "source": f"Dyndata01_row_{row_idx:04d}",
                "row_index": row_idx,
                "step_index": int(row["step_index"]) if "step_index" in row else row_idx,
                "phase": row.get("phase", ""),
                "planned_time_s": row.get("planned_time_s", ""),
                "actual_time_s": row.get("actual_time_s", ""),
                "raw_u_mm": raw_u_mm,
                "cable_target_mm": cable_target_mm,
                "motor_sent_mm": _motor_sent_from_row(row),
                "q_initial": q_guess.copy(),
                "q_est": q_est,
                "feedback_accepted": accepted,
                "cable_residual_norm": cable_residual_norm,
                "cable_weight": cable_weight,
                "endeff_residual_norm": endeff_residual_norm,
                "endeff_weight": endeff_weight,
                "regularization_weight": regularization_weight,
                "feedback_time_s": elapsed,
                "pcs_endeff_position_mm": pcs_endeff_position_mm,
                "pcs_endeff_displacement_mm": pcs_endeff_displacement_mm,
                "cable_length_m": cable_length_m,
                "cable_delta_mm": cable_delta_mm,
                "raw_u_residual_mm": cable_delta_mm - cable_target_mm,
                "motor_order_cable_delta_mm": motor_order_delta_mm,
                "motor_sent_residual_mm": motor_order_delta_mm - _motor_sent_from_row(row),
                "measurement_endeff_mm": endeff_target_mm,
            }
        )
        q_guess = q_est

    return results


def write_csv(output_path: Path, results: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    q_count = np.asarray(results[0]["q_est"]).reshape(-1).size
    headers = [
        "row_index",
        "step_index",
        "phase",
        "planned_time_s",
        "actual_time_s",
        "feedback_accepted",
        "cable_residual_norm",
        "cable_weight",
        "endeff_residual_norm",
        "endeff_weight",
        "regularization_weight",
        "feedback_time_s",
    ]
    headers.extend([f"q_{idx + 1}" for idx in range(q_count)])
    headers.extend([f"measurement_endeff_{axis}_mm" for axis in AXES])
    headers.extend([f"pcs_endeff_position_{axis}_mm" for axis in AXES])
    headers.extend([f"pcs_endeff_displacement_{axis}_mm" for axis in AXES])
    headers.extend([f"raw_u{idx + 1}_mm" for idx in range(MOTOR_COUNT)])
    headers.extend([f"cable_target_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"cable_delta_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"raw_u_residual_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"motor_order_cable_delta_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"motor_sent_u{idx + 1}_mm" for idx in range(MOTOR_COUNT)])
    headers.extend([f"motor_sent_residual_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for result in results:
            row = {key: result[key] for key in headers[:12]}
            for idx, value in enumerate(np.asarray(result["q_est"], dtype=float).reshape(-1)):
                row[f"q_{idx + 1}"] = float(value)
            for axis, value in zip(AXES, result["measurement_endeff_mm"]):
                row[f"measurement_endeff_{axis}_mm"] = float(value)
            for axis, value in zip(AXES, result["pcs_endeff_position_mm"]):
                row[f"pcs_endeff_position_{axis}_mm"] = float(value)
            for axis, value in zip(AXES, result["pcs_endeff_displacement_mm"]):
                row[f"pcs_endeff_displacement_{axis}_mm"] = float(value)
            for idx, value in enumerate(result["raw_u_mm"]):
                row[f"raw_u{idx + 1}_mm"] = float(value)
            for prefix in ("cable_target_mm", "cable_delta_mm", "raw_u_residual_mm", "motor_order_cable_delta_mm"):
                for idx, value in enumerate(np.asarray(result[prefix], dtype=float).reshape(6)):
                    row[f"{prefix}_{idx + 1}"] = float(value)
            for idx, value in enumerate(result["motor_sent_mm"]):
                row[f"motor_sent_u{idx + 1}_mm"] = float(value)
            for idx, value in enumerate(result["motor_sent_residual_mm"]):
                row[f"motor_sent_residual_mm_{idx + 1}"] = float(value)
            writer.writerow(row)


def write_log(output_path: Path, results: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = sum(bool(result["feedback_accepted"]) for result in results)
    time_s = _integration_time_s(results)
    cable_errors = np.asarray([result["raw_u_residual_mm"] for result in results], dtype=float)
    endeff_errors = np.asarray(
        [result["pcs_endeff_displacement_mm"] - result["measurement_endeff_mm"] for result in results],
        dtype=float,
    )
    cable_ise, cable_ise_total = _integral_squared_error(time_s, cable_errors)
    endeff_ise, endeff_ise_total = _integral_squared_error(time_s, endeff_errors)
    lines = [
        "DynamicVerification PCS estimation summary",
        f"Samples: {len(results)}",
        f"Accepted: {accepted_count}/{len(results)}",
        f"Cable residual weight: {results[0]['cable_weight'] if results else ''}",
        f"Endeff residual weight: {results[0]['endeff_weight'] if results else ''}",
        f"Regularization residual weight: {results[0]['regularization_weight'] if results else ''}",
        "",
        "Integral squared error summary",
        f"Integration time range s: {time_s[0]:.8g} to {time_s[-1]:.8g}" if time_s.size else "Integration time range s: ",
        "Cable target-vs-delta ISE mm^2*s: " + _format_vector(cable_ise, 8),
        f"Cable target-vs-delta total ISE mm^2*s: {cable_ise_total:.8g}",
        "Endeff measured-vs-PCS ISE mm^2*s: " + _format_vector(endeff_ise, 8),
        f"Endeff measured-vs-PCS total ISE mm^2*s: {endeff_ise_total:.8g}",
        "",
    ]
    for result in results:
        lines.append(
            f"{result['source']}: t={result['actual_time_s']}, accepted={result['feedback_accepted']}, "
            f"cable_residual={result['cable_residual_norm']:.8g}, "
            f"endeff_residual={result['endeff_residual_norm']:.8g}"
        )
        lines.append(f"  q_est: {_format_vector(result['q_est'])}")
        lines.append(f"  cable_target_mm: {_format_vector(result['cable_target_mm'], 6)}")
        lines.append(f"  cable_delta_mm: {_format_vector(result['cable_delta_mm'], 6)}")
        lines.append(f"  pcs_endeff_displacement_mm: {_format_vector(result['pcs_endeff_displacement_mm'], 6)}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Estimate dynamic states with the PCS model for Dyndata01.")
    parser.add_argument("--input", type=Path, default=script_dir / "DynData" / "Dyndata01.csv")
    parser.add_argument("--output", type=Path, default=script_dir / "results" / "pcs_estimates.npy")
    parser.add_argument("--csv", type=Path, default=script_dir / "results" / "pcs_estimates.csv")
    parser.add_argument("--log", type=Path, default=script_dir / "results" / "pcs_estimates_log.txt")
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--cable-weight", type=float, default=DEFAULT_CABLE_WEIGHT)
    parser.add_argument("--endeff-weight", type=float, default=DEFAULT_ENDEFF_WEIGHT)
    parser.add_argument("--regularization-weight", type=float, default=DEFAULT_REGULARIZATION_WEIGHT)
    parser.add_argument("--independent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("cable_weight", "endeff_weight", "regularization_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    rows = _load_csv_rows(args.input)
    results = estimate_rows(
        rows,
        sequential=not args.independent,
        max_nfev=args.max_nfev,
        cable_weight=args.cable_weight,
        endeff_weight=args.endeff_weight,
        regularization_weight=args.regularization_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, results, allow_pickle=True)
    write_csv(args.csv, results)
    write_log(args.log, results)
    accepted_count = sum(bool(result["feedback_accepted"]) for result in results)
    print(f"Estimated {len(results)} PCS rows; accepted {accepted_count}/{len(results)}.")
    print(f"Saved PCS estimates to {args.output}")
    print(f"Saved CSV summary to {args.csv}")
    print(f"Saved log to {args.log}")


if __name__ == "__main__":
    main()
