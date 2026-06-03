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

from Main_demo_SSM import (
    MOTOR_FACTOR,
    POS_END,
    build_default_config,
    build_initial_feedback_state,
    build_ssm_maps,
)
from controller.cable_constraints import cable_length_double_part
from controller.ssm_controller import SsmMpcController
from dynamics.SSM.matlab_data import EIGHT_TRAJ_DIR, ROM_DIR, get_field, load_rom_data
from dynamics.SSM.ssm_dynamics import reduced_to_full
from measure.motion_capture import MotionReference
from zmq_message.zmq_control import reorder_motor_positions


MEASUREMENT_NAMES = ("endeff", "mideff", "checkpoint")
AXES = ("x", "y", "z")
RAW_U_COLUMNS = tuple(f"raw_u{idx}_mm" for idx in range(1, 7))
MOTOR_COLUMNS = tuple(f"motor_sent_u{idx}_mm" for idx in range(1, 7))
MOTOR_COUNT = 6

# Least-squares residual weights. Tune these defaults here, or override them with CLI arguments.
# DEFAULT_CABLE_WEIGHT = 1.0
# DEFAULT_ENDEFF_WEIGHT = 0.3 # 0.11
# DEFAULT_REGULARIZATION_WEIGHT = 1.25

# DEFAULT_CABLE_WEIGHT = 2.5
# DEFAULT_ENDEFF_WEIGHT = 1 # 0.11
# DEFAULT_REGULARIZATION_WEIGHT = 1e-3

DEFAULT_CABLE_WEIGHT = 2.5
DEFAULT_ENDEFF_WEIGHT = 0.25 # 0.11
DEFAULT_REGULARIZATION_WEIGHT = 1e-3

def swap_cable_pairs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(MOTOR_COUNT)
    # return values[[0, 2, 1, 3, 5, 4]]
    return values[[0, 1, 2, 3, 4, 5]]

def _load_npy_object(path: Path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    return data.tolist()


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


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


def _node_index_from_dofs(dofs_1based: np.ndarray) -> int:
    dofs = np.asarray(dofs_1based, dtype=int).reshape(-1)
    node_indices = np.unique((dofs - 1) // 3)
    if node_indices.size != 1:
        raise ValueError(f"Expected one node index from measurement DOFs, got {node_indices}")
    return int(node_indices[0])


def _model_default_reference_positions_mm(rom_data: dict) -> dict[str, np.ndarray]:
    nodes_xyz = np.asarray(rom_data["nodes_coord"], dtype=float).T
    mid_index = _node_index_from_dofs(np.asarray(rom_data["outdof_half"], dtype=int).reshape(-1))
    mid_offset_mm = (nodes_xyz[mid_index] - POS_END) * 1000.0
    return {
        "endeff": np.zeros(3),
        "mideff": mid_offset_mm,
        "checkpoint": mid_offset_mm,
    }


def load_reference_positions(reference_path: Path | None, rom_data: dict) -> dict[str, np.ndarray]:
    if reference_path is None or not reference_path.exists():
        return _model_default_reference_positions_mm(rom_data)

    loaded = _as_list(_load_npy_object(reference_path))
    for item in loaded:
        if isinstance(item, dict) and "reference_initial_positions_mm" in item:
            positions = item["reference_initial_positions_mm"]
            return {name: np.asarray(positions[name], dtype=float).reshape(3) for name in MEASUREMENT_NAMES}
        if isinstance(item, dict) and "initial_reference_positions_mm" in item:
            positions = item["initial_reference_positions_mm"]
            return {name: np.asarray(positions[name], dtype=float).reshape(3) for name in MEASUREMENT_NAMES}

    raise KeyError(
        f"{reference_path} does not contain reference_initial_positions_mm or initial_reference_positions_mm"
    )


def _reference_from_positions(initial_positions_mm: dict[str, np.ndarray]) -> MotionReference:
    return MotionReference(
        end_effector=np.asarray(initial_positions_mm["endeff"], dtype=float).reshape(3),
        mid_effector=np.asarray(initial_positions_mm["mideff"], dtype=float).reshape(3),
        checkpoint=np.asarray(initial_positions_mm["checkpoint"], dtype=float).reshape(3),
        base_xyz=np.zeros(3),
        base_to_global=np.eye(3),
    )


def _make_controller(rom_data: dict, reference: MotionReference) -> SsmMpcController:
    config = build_default_config(rom_data)
    x_num = np.asarray(get_field(rom_data["auData"], "lamd")).reshape(-1).size
    w0_for_out, w0_for_cables, w0_for_feedback = build_ssm_maps(rom_data, reference)
    return SsmMpcController(
        config=config,
        au_data=rom_data["auData"],
        w0_for_out=w0_for_out,
        w0_for_cables=w0_for_cables,
        w0_for_feedback=w0_for_feedback,
        feedback_state=build_initial_feedback_state(x_num, MOTOR_COUNT, config.Np),
        factor=MOTOR_FACTOR,
        feedback_enabled=True,
        save_feedback_each_step=False,
    )


def _measurement_from_row(row: dict[str, object]) -> np.ndarray:
    values = []
    for name in MEASUREMENT_NAMES:
        for axis in AXES:
            values.append(float(row[f"{name}_disp_{axis}_ref_mm"]))
    return np.asarray(values, dtype=float)


def _motor_sent_from_row(row: dict[str, object]) -> np.ndarray:
    return np.asarray([float(row[column]) for column in MOTOR_COLUMNS], dtype=float)


def _raw_u_from_row(row: dict[str, object]) -> np.ndarray:
    return np.asarray([float(row[column]) for column in RAW_U_COLUMNS], dtype=float)


def _measurement_residual(
    raw_u_mm: np.ndarray,
    measurement_mm: np.ndarray,
    pk: np.ndarray,
    controller: SsmMpcController,
    pk_guess: np.ndarray,
    *,
    cable_weight: float,
    endeff_weight: float,
    regularization_weight: float,
) -> np.ndarray:
    cable_displacement = reduced_to_full(pk, controller.w0_for_cables, None, 1).reshape(-1)
    cable_length_m = cable_length_double_part(cable_displacement, controller.config.Node_List)
    cable_delta_m = cable_length_m - controller.config.dist0
    cable_target_m = swap_cable_pairs(raw_u_mm) / 1000.0
    cable_residual = np.sqrt(cable_weight) * (cable_delta_m - cable_target_m)

    y_cal = reduced_to_full(pk, controller.w0_for_feedback, None, 1).reshape(-1)
    endeff_target_mm = np.asarray(measurement_mm, dtype=float).reshape(3, 3)[0].copy()
    endeff_target_mm[0] *= -1.0
    endeff_residual = np.sqrt(endeff_weight) * (y_cal[:3] - endeff_target_mm / 1000.0)

    regularization_residual = np.sqrt(regularization_weight) * (pk - pk_guess)
    return np.concatenate([cable_residual, endeff_residual, regularization_residual])


def estimate_pk(
    controller: SsmMpcController,
    pk_guess: np.ndarray,
    raw_u_mm: np.ndarray,
    measurement_mm: np.ndarray,
    *,
    max_nfev: int,
    cable_weight: float,
    endeff_weight: float,
    regularization_weight: float,
) -> tuple[np.ndarray, bool, float, float, float]:
    start_time = time.perf_counter()
    pk_guess = np.asarray(pk_guess, dtype=float).reshape(-1)
    result = least_squares(
        lambda pk: _measurement_residual(
            raw_u_mm,
            measurement_mm,
            pk,
            controller,
            pk_guess,
            cable_weight=cable_weight,
            endeff_weight=endeff_weight,
            regularization_weight=regularization_weight,
        ),
        pk_guess,
        method="lm",
        max_nfev=max_nfev,
    )
    feedback_time = time.perf_counter() - start_time
    residual = _measurement_residual(
        raw_u_mm,
        measurement_mm,
        result.x,
        controller,
        pk_guess,
        cable_weight=cable_weight,
        endeff_weight=endeff_weight,
        regularization_weight=regularization_weight,
    )
    cable_residual_norm = float(np.linalg.norm(residual[:MOTOR_COUNT], 2))
    endeff_residual_norm = float(np.linalg.norm(residual[MOTOR_COUNT : MOTOR_COUNT + 3], 2))
    return result.x, cable_residual_norm < 1e-2, cable_residual_norm, endeff_residual_norm, feedback_time


def _initial_model_positions_m(reference: MotionReference) -> dict[str, np.ndarray]:
    endeff0 = np.asarray(POS_END, dtype=float).reshape(3)
    return {
        "endeff": endeff0,
        "mideff": ((np.asarray(reference.mid_effector, dtype=float) - reference.end_effector) / 1000.0) + endeff0,
        "checkpoint": ((np.asarray(reference.checkpoint, dtype=float) - reference.end_effector) / 1000.0) + endeff0,
    }


def calculate_model_outputs(
    controller: SsmMpcController,
    reference: MotionReference,
    pk_est: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    feedback_displacement_m = reduced_to_full(pk_est, controller.w0_for_feedback, None, 1).reshape(3, 3)
    initial_positions_m = _initial_model_positions_m(reference)
    displacements = {
        name: feedback_displacement_m[idx] for idx, name in enumerate(MEASUREMENT_NAMES)
    }
    positions = {
        name: initial_positions_m[name] + displacements[name] for name in MEASUREMENT_NAMES
    }
    return positions, displacements


def calculate_cable_outputs(
    controller: SsmMpcController,
    pk_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cable_displacement = reduced_to_full(pk_est, controller.w0_for_cables, None, 1).reshape(-1)
    cable_length_m = cable_length_double_part(cable_displacement, controller.config.Node_List)
    cable_delta_mm = (cable_length_m - controller.config.dist0) * 1000.0
    motor_order_delta_mm = np.asarray(reorder_motor_positions(cable_delta_mm), dtype=float)
    motor_command_mm = motor_order_delta_mm * MOTOR_FACTOR
    return cable_length_m, cable_delta_mm, motor_order_delta_mm, motor_command_mm


def estimate_dynamic_rows(
    rows: list[dict[str, object]],
    *,
    reference_path: Path | None,
    sequential: bool,
    max_nfev: int,
    cable_weight: float,
    endeff_weight: float,
    regularization_weight: float,
) -> list[dict[str, object]]:
    rom_data = load_rom_data(ROM_DIR, traj_dir=EIGHT_TRAJ_DIR)
    initial_positions_mm = load_reference_positions(reference_path, rom_data)
    reference = _reference_from_positions(initial_positions_mm)
    controller = _make_controller(rom_data, reference)
    pk_guess = np.zeros(controller.x_num)
    results = []

    for row_idx, row in enumerate(rows):
        if not sequential:
            pk_guess = np.zeros(controller.x_num)

        measurement_mm = _measurement_from_row(row)
        raw_u_mm = _raw_u_from_row(row)
        cable_target_mm = swap_cable_pairs(raw_u_mm)
        motor_sent_mm = _motor_sent_from_row(row)
        pk_est, accepted, residual_norm, endeff_residual_norm, feedback_time = estimate_pk(
            controller,
            pk_guess,
            raw_u_mm,
            measurement_mm,
            max_nfev=max_nfev,
            cable_weight=cable_weight,
            endeff_weight=endeff_weight,
            regularization_weight=regularization_weight,
        )
        model_positions_m, model_displacements_m = calculate_model_outputs(controller, reference, pk_est)
        cable_length_m, cable_delta_mm, motor_order_delta_mm, motor_command_mm = calculate_cable_outputs(
            controller,
            pk_est,
        )

        results.append(
            {
                "source": f"Dyndata01_row_{row_idx:04d}",
                "row_index": row_idx,
                "step_index": int(row["step_index"]) if "step_index" in row else row_idx,
                "phase": row.get("phase", ""),
                "planned_time_s": row.get("planned_time_s", ""),
                "actual_time_s": row.get("actual_time_s", ""),
                "measurement_order": MEASUREMENT_NAMES,
                "measurement_mm": measurement_mm,
                "raw_u_mm": raw_u_mm,
                "cable_target_mm": cable_target_mm,
                "motor_sent_mm": motor_sent_mm,
                "pk_initial": pk_guess.copy(),
                "pk_est": pk_est,
                "feedback_accepted": accepted,
                "cable_residual_norm": residual_norm,
                "cable_weight": cable_weight,
                "endeff_residual_norm": endeff_residual_norm,
                "endeff_weight": endeff_weight,
                "regularization_weight": regularization_weight,
                "feedback_residual_norm": residual_norm,
                "feedback_time_s": feedback_time,
                "ssm_position_m": model_positions_m,
                "ssm_displacement_m": model_displacements_m,
                "cable_length_m": cable_length_m,
                "cable_delta_mm": cable_delta_mm,
                "motor_order_cable_delta_mm": motor_order_delta_mm,
                "motor_command_mm": motor_command_mm,
                "raw_u_residual_mm": cable_delta_mm - cable_target_mm,
                "motor_sent_residual_mm": motor_command_mm - motor_sent_mm,
                "reference_initial_positions_mm": initial_positions_mm,
            }
        )
        pk_guess = pk_est

    return results


def _format_vector(values: np.ndarray, precision: int = 8) -> str:
    return "[" + ", ".join(f"{value:.{precision}g}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def _integration_time_s(results: list[dict[str, object]]) -> np.ndarray:
    values = []
    for idx, result in enumerate(results):
        value = result.get("actual_time_s", "")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float(idx))
    time_s = np.asarray(values, dtype=float)
    if time_s.size > 1 and np.all(np.diff(time_s) >= 0):
        return time_s
    return np.arange(len(results), dtype=float)


def _integral_squared_error(time_s: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, float]:
    errors = np.asarray(errors, dtype=float)
    squared = errors**2
    if time_s.size > 1:
        per_axis = np.trapezoid(squared, x=time_s, axis=0)
    else:
        per_axis = np.zeros(squared.shape[1], dtype=float)
    return per_axis, float(np.sum(per_axis))


def _summary_integral_errors(results: list[dict[str, object]]) -> dict[str, object]:
    time_s = _integration_time_s(results)
    cable_errors = np.asarray(
        [np.asarray(result["raw_u_residual_mm"], dtype=float).reshape(6) for result in results],
        dtype=float,
    )
    endeff_errors = []
    for result in results:
        measured = np.asarray(result["measurement_mm"], dtype=float).reshape(3, 3)[0].copy()
        measured[0] *= -1.0
        ssm = np.asarray(result["ssm_displacement_m"]["endeff"], dtype=float).reshape(3) * 1000.0
        endeff_errors.append(ssm - measured)
    endeff_errors = np.asarray(endeff_errors, dtype=float)

    cable_ise, cable_ise_total = _integral_squared_error(time_s, cable_errors)
    endeff_ise, endeff_ise_total = _integral_squared_error(time_s, endeff_errors)
    return {
        "time_start_s": float(time_s[0]) if time_s.size else 0.0,
        "time_end_s": float(time_s[-1]) if time_s.size else 0.0,
        "cable_ise_mm2_s": cable_ise,
        "cable_ise_total_mm2_s": cable_ise_total,
        "endeff_ise_mm2_s": endeff_ise,
        "endeff_ise_total_mm2_s": endeff_ise_total,
    }


def write_log(output_path: Path, results: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = sum(bool(result["feedback_accepted"]) for result in results)
    integral_errors = _summary_integral_errors(results) if results else None
    lines = [
        "DynamicVerification pk estimation summary",
        f"Samples: {len(results)}",
        f"Accepted: {accepted_count}/{len(results)}",
        f"Cable residual weight: {results[0]['cable_weight'] if results else ''}",
        f"Endeff residual weight: {results[0]['endeff_weight'] if results else ''}",
        f"Regularization residual weight: {results[0]['regularization_weight'] if results else ''}",
        "",
    ]
    if integral_errors is not None:
        lines.extend(
            [
                "Integral squared error summary",
                f"Integration time range s: {integral_errors['time_start_s']:.8g} to {integral_errors['time_end_s']:.8g}",
                "Cable target-vs-delta ISE mm^2*s: "
                + _format_vector(integral_errors["cable_ise_mm2_s"], 8),
                f"Cable target-vs-delta total ISE mm^2*s: {integral_errors['cable_ise_total_mm2_s']:.8g}",
                "Endeff measured-vs-SSM ISE mm^2*s: "
                + _format_vector(integral_errors["endeff_ise_mm2_s"], 8),
                f"Endeff measured-vs-SSM total ISE mm^2*s: {integral_errors['endeff_ise_total_mm2_s']:.8g}",
                "",
            ]
        )
    for result in results:
        lines.append(
            f"{result['source']}: t={result['actual_time_s']}, accepted={result['feedback_accepted']}, "
            f"cable_residual={result['cable_residual_norm']:.8g}, "
            f"endeff_residual={result['endeff_residual_norm']:.8g}"
        )
        lines.append(f"  pk_est: {_format_vector(result['pk_est'])}")
        lines.append(f"  raw_u_mm: {_format_vector(result['raw_u_mm'], 6)}")
        lines.append(f"  cable_target_mm: {_format_vector(result['cable_target_mm'], 6)}")
        lines.append(f"  cable_delta_mm: {_format_vector(result['cable_delta_mm'], 6)}")
        lines.append(f"  raw_u_residual_mm: {_format_vector(result['raw_u_residual_mm'], 6)}")
        lines.append(f"  motor_command_mm: {_format_vector(result['motor_command_mm'], 6)}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(output_path: Path, results: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return

    pk_count = np.asarray(results[0]["pk_est"]).reshape(-1).size
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
    headers.extend([f"pk_{idx + 1}" for idx in range(pk_count)])
    headers.extend([f"measurement_{name}_{axis}_mm" for name in MEASUREMENT_NAMES for axis in AXES])
    headers.extend([f"ssm_{name}_position_{axis}_mm" for name in MEASUREMENT_NAMES for axis in AXES])
    headers.extend([f"ssm_{name}_displacement_{axis}_mm" for name in MEASUREMENT_NAMES for axis in AXES])
    headers.extend([f"raw_u{idx + 1}_mm" for idx in range(MOTOR_COUNT)])
    headers.extend([f"cable_target_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"motor_sent_u{idx + 1}_mm" for idx in range(MOTOR_COUNT)])
    headers.extend([f"cable_delta_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"raw_u_residual_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"motor_order_cable_delta_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"motor_command_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])
    headers.extend([f"motor_sent_residual_mm_{idx + 1}" for idx in range(MOTOR_COUNT)])

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for result in results:
            row = {
                "row_index": result["row_index"],
                "step_index": result["step_index"],
                "phase": result["phase"],
                "planned_time_s": result["planned_time_s"],
                "actual_time_s": result["actual_time_s"],
                "feedback_accepted": result["feedback_accepted"],
                "cable_residual_norm": result["cable_residual_norm"],
                "cable_weight": result["cable_weight"],
                "endeff_residual_norm": result["endeff_residual_norm"],
                "endeff_weight": result["endeff_weight"],
                "regularization_weight": result["regularization_weight"],
                "feedback_time_s": result["feedback_time_s"],
            }
            for idx, value in enumerate(np.asarray(result["pk_est"], dtype=float).reshape(-1)):
                row[f"pk_{idx + 1}"] = float(value)
            for name_idx, name in enumerate(MEASUREMENT_NAMES):
                measured = np.asarray(result["measurement_mm"], dtype=float).reshape(3, 3)[name_idx]
                position_mm = np.asarray(result["ssm_position_m"][name], dtype=float).reshape(3) * 1000.0
                displacement_mm = np.asarray(result["ssm_displacement_m"][name], dtype=float).reshape(3) * 1000.0
                for axis, value in zip(AXES, measured):
                    row[f"measurement_{name}_{axis}_mm"] = float(value)
                for axis, value in zip(AXES, position_mm):
                    row[f"ssm_{name}_position_{axis}_mm"] = float(value)
                for axis, value in zip(AXES, displacement_mm):
                    row[f"ssm_{name}_displacement_{axis}_mm"] = float(value)
            for idx, value in enumerate(np.asarray(result["raw_u_mm"], dtype=float).reshape(6)):
                row[f"raw_u{idx + 1}_mm"] = float(value)
            for idx, value in enumerate(np.asarray(result["cable_target_mm"], dtype=float).reshape(6)):
                row[f"cable_target_mm_{idx + 1}"] = float(value)
            for idx, value in enumerate(np.asarray(result["motor_sent_mm"], dtype=float).reshape(6)):
                row[f"motor_sent_u{idx + 1}_mm"] = float(value)
            for prefix in (
                "cable_delta_mm",
                "raw_u_residual_mm",
                "motor_order_cable_delta_mm",
                "motor_command_mm",
                "motor_sent_residual_mm",
            ):
                for idx, value in enumerate(np.asarray(result[prefix], dtype=float).reshape(6)):
                    row[f"{prefix}_{idx + 1}"] = float(value)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_reference = (
        REPO_ROOT
        / "StaticVerification"
        / "results"
        / "Static_data"
        / "static_verification_max_displacement_measurements.npy"
    )
    parser = argparse.ArgumentParser(
        description="Estimate SSM reduced states pk for each row of DynamicVerification/DynData/Dyndata01.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "DynData" / "Dyndata01.csv",
        help="Input dynamic CSV.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=default_reference,
        help=(
            "Optional npy containing reference_initial_positions_mm or initial_reference_positions_mm. "
            "Used to select the SSM checkpoint node and report spatial positions."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / "dynamic_pk_estimates.npy",
        help="Output npy containing full per-row estimation data.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=script_dir / "results" / "dynamic_pk_estimates.csv",
        help="Output CSV summary.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=script_dir / "results" / "dynamic_pk_estimates_log.txt",
        help="Output text log.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=20,
        help="Maximum least-squares function evaluations for each row.",
    )
    parser.add_argument(
        "--cable-weight",
        type=float,
        default=DEFAULT_CABLE_WEIGHT,
        help="Least-squares weight for cable-length residual.",
    )
    parser.add_argument(
        "--endeff-weight",
        type=float,
        default=DEFAULT_ENDEFF_WEIGHT,
        help=(
            "Small least-squares weight for endeff displacement residual. "
            "Set to 0 to use cable length and regularization only."
        ),
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=DEFAULT_REGULARIZATION_WEIGHT,
        help="Least-squares weight for pk regularization residual.",
    )
    parser.add_argument(
        "--independent",
        action="store_true",
        help="Estimate each row from zero instead of using the previous pk as the next initial guess.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("cable_weight", "endeff_weight", "regularization_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    rows = _load_csv_rows(args.input)
    if not rows:
        raise ValueError(f"No data rows found in {args.input}")

    results = estimate_dynamic_rows(
        rows,
        reference_path=args.reference,
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
    print(f"Estimated {len(results)} dynamic rows; accepted {accepted_count}/{len(results)}.")
    print(f"Saved pk estimates to {args.output}")
    print(f"Saved CSV summary to {args.csv}")
    print(f"Saved log to {args.log}")


if __name__ == "__main__":
    main()
