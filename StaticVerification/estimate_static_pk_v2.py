from __future__ import annotations

import argparse
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Main_demo_SSM import (
    MOTOR_FACTOR,
    build_default_config,
    build_initial_feedback_state,
    build_ssm_maps,
)
from controller.cable_constraints import cable_length_double_part
from controller.ssm_controller import SsmMpcController
from dynamics.SSM.ssm_dynamics import reduced_to_full
from dynamics.SSM.matlab_data import EIGHT_TRAJ_DIR, ROM_DIR, get_field, load_rom_data
from measure.motion_capture import MotionReference


CABLE_ANGLE_PER_LENGTH = 2.29183118


def swap_cable_pairs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(6)
    return values[[0, 2, 1, 3, 5, 4]]


def _load_npy_object(path: Path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    return data.tolist()


def _read_xlsx_rows(path: Path) -> list[list[object]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as xlsx:
        shared_strings = []
        if "xl/sharedStrings.xml" in xlsx.namelist():
            root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))

        sheet_names = sorted(
            name for name in xlsx.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheet_names:
            raise ValueError(f"No worksheets found in {path}")

        root = ET.fromstring(xlsx.read(sheet_names[0]))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                value = cell.find("a:v", ns)
                if value is None:
                    values.append(None)
                elif cell.attrib.get("t") == "s":
                    values.append(shared_strings[int(value.text)])
                else:
                    text = value.text
                    try:
                        values.append(float(text))
                    except (TypeError, ValueError):
                        values.append(text)
            rows.append(values)
    return rows


def load_cable_targets(path: Path, samples: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    rows = _read_xlsx_rows(path)
    data_rows = [row for row in rows[1:] if len(row) >= 8]
    if len(data_rows) < len(samples):
        raise ValueError(f"{path} has {len(data_rows)} cable-length rows, but {len(samples)} samples were provided.")

    targets = {}
    for sample, row in zip(samples, data_rows):
        target_angle_raw = np.asarray(row[2:8], dtype=float).reshape(6)
        target_angle = swap_cable_pairs(target_angle_raw)
        targets[sample["source"]] = {
            "angle_raw": target_angle_raw,
            "angle": target_angle,
            "length_mm": target_angle / CABLE_ANGLE_PER_LENGTH,
        }
    return targets


def _reference_from_sample(sample: dict) -> MotionReference:
    initial_positions = sample["reference_initial_positions_mm"]
    return MotionReference(
        end_effector=np.asarray(initial_positions["endeff"], dtype=float).reshape(3),
        mid_effector=np.asarray(initial_positions["mideff"], dtype=float).reshape(3),
        checkpoint=np.asarray(initial_positions["checkpoint"], dtype=float).reshape(3),
        base_xyz=np.zeros(3),
        base_to_global=np.eye(3),
    )


def feedback_residual_v2(
    yk_mm: np.ndarray,
    cable_target_length_mm: np.ndarray,
    pk: np.ndarray,
    controller: SsmMpcController,
    p_guess: np.ndarray,
    *,
    cable_weight: float,
) -> np.ndarray:
    y_cal = reduced_to_full(pk, controller.w0_for_feedback, None, 1).reshape(-1)
    measurement_residual = y_cal - np.asarray(yk_mm, dtype=float).reshape(9) / 1000.0

    cable_displacement = reduced_to_full(pk, controller.w0_for_cables, None, 1).reshape(-1)
    cable_length = cable_length_double_part(cable_displacement, controller.config.Node_List)
    cable_delta_m = cable_length - controller.config.dist0
    cable_target_m = np.asarray(cable_target_length_mm, dtype=float).reshape(6) / 1000.0
    cable_residual = np.sqrt(cable_weight) * (cable_delta_m - cable_target_m)

    regularization_residual = np.sqrt(controller.config.opt_W) * (pk - p_guess)
    return np.concatenate([measurement_residual, cable_residual, regularization_residual])


def apply_measurement_feedback_v2(
    controller: SsmMpcController,
    pk: np.ndarray,
    measurement_mm: np.ndarray,
    cable_target_length_mm: np.ndarray,
    *,
    cable_weight: float,
    max_nfev: int,
) -> tuple[np.ndarray, bool, float, float, float, np.ndarray, np.ndarray]:
    start_time = time.perf_counter()
    pk = np.asarray(pk, dtype=float).reshape(-1)
    result = least_squares(
        lambda p: feedback_residual_v2(
            measurement_mm,
            cable_target_length_mm,
            p,
            controller,
            pk,
            cable_weight=cable_weight,
        ),
        pk,
        method="lm",
        max_nfev=max_nfev,
    )
    feedback_time = time.perf_counter() - start_time
    residual = feedback_residual_v2(
        measurement_mm,
        cable_target_length_mm,
        result.x,
        controller,
        pk,
        cable_weight=cable_weight,
    )
    measurement_residual_norm = float(np.linalg.norm(residual[:9], 2))
    cable_residual_norm = float(np.linalg.norm(residual[9:15], 2))
    accepted = measurement_residual_norm < 1e-2
    cable_displacement = reduced_to_full(result.x, controller.w0_for_cables, None, 1).reshape(-1)
    cable_length_m = cable_length_double_part(cable_displacement, controller.config.Node_List)
    cable_delta_mm = (cable_length_m - controller.config.dist0) * 1000.0
    return result.x, accepted, measurement_residual_norm, cable_residual_norm, feedback_time, cable_length_m, cable_delta_mm


def _make_controller(rom_data, reference: MotionReference) -> SsmMpcController:
    config = build_default_config(rom_data)
    x_num = np.asarray(get_field(rom_data["auData"], "lamd")).reshape(-1).size
    w0_for_out, w0_for_cables, w0_for_feedback = build_ssm_maps(rom_data, reference)
    return SsmMpcController(
        config=config,
        au_data=rom_data["auData"],
        w0_for_out=w0_for_out,
        w0_for_cables=w0_for_cables,
        w0_for_feedback=w0_for_feedback,
        feedback_state=build_initial_feedback_state(x_num, 6, config.Np),
        factor=MOTOR_FACTOR,
        feedback_enabled=True,
        save_feedback_each_step=False,
    )


def estimate_pk_for_samples(
    samples: list[dict],
    cable_targets: dict[str, dict[str, np.ndarray]],
    *,
    sequential: bool = False,
    cable_weight: float = 0.5,
    max_nfev: int = 50,
) -> list[dict]:
    rom_data = load_rom_data(ROM_DIR, traj_dir=EIGHT_TRAJ_DIR)
    results = []
    pk_guess = None

    for sample in samples:
        reference = _reference_from_sample(sample)
        controller = _make_controller(rom_data, reference)
        if pk_guess is None or not sequential:
            pk_guess = np.zeros(controller.x_num)

        measurement_mm = np.asarray(sample["measurement_mm"], dtype=float).reshape(9)
        cable_target_angle_raw = np.asarray(cable_targets[sample["source"]]["angle_raw"], dtype=float).reshape(6)
        cable_target_angle = np.asarray(cable_targets[sample["source"]]["angle"], dtype=float).reshape(6)
        cable_target_length_mm = np.asarray(cable_targets[sample["source"]]["length_mm"], dtype=float).reshape(6)
        (
            pk_est,
            accepted,
            measurement_residual_norm,
            cable_residual_norm,
            feedback_time,
            cable_estimated_length_m,
            cable_estimated_delta_mm,
        ) = apply_measurement_feedback_v2(
            controller,
            pk_guess,
            measurement_mm,
            cable_target_length_mm,
            cable_weight=cable_weight,
            max_nfev=max_nfev,
        )
        result = {
            "source": sample["source"],
            "measurement_order": sample["measurement_order"],
            "measurement_mm": measurement_mm,
            "cable_target_angle_raw": cable_target_angle_raw,
            "cable_target_angle": cable_target_angle,
            "cable_target_length_mm": cable_target_length_mm,
            "cable_estimated_length_m": cable_estimated_length_m,
            "cable_estimated_delta_mm": cable_estimated_delta_mm,
            "pk_initial": pk_guess,
            "pk_est": pk_est,
            "feedback_accepted": accepted,
            "feedback_residual_norm": measurement_residual_norm,
            "cable_residual_norm": cable_residual_norm,
            "feedback_time_s": feedback_time,
            "cable_weight": cable_weight,
        }
        results.append(result)
        if sequential:
            pk_guess = pk_est

    return results


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.8g}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def write_estimation_log(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["StaticVerification pk estimation summary v2", ""]
    for result in results:
        lines.append(f"Source: {result['source']}")
        lines.append(f"Measurement order: {', '.join(result['measurement_order'])}")
        lines.append(f"Measurement mm: {_format_vector(result['measurement_mm'])}")
        lines.append(f"Cable target angle raw: {_format_vector(result['cable_target_angle_raw'])}")
        lines.append(f"Cable target angle: {_format_vector(result['cable_target_angle'])}")
        lines.append(f"Cable target length mm: {_format_vector(result['cable_target_length_mm'])}")
        lines.append(f"Cable estimated delta mm: {_format_vector(result['cable_estimated_delta_mm'])}")
        lines.append(f"Cable estimated length m: {_format_vector(result['cable_estimated_length_m'])}")
        lines.append(f"Feedback accepted: {result['feedback_accepted']}")
        lines.append(f"Measurement residual norm: {result['feedback_residual_norm']}")
        lines.append(f"Cable residual norm: {result['cable_residual_norm']}")
        lines.append(f"Cable weight: {result['cable_weight']}")
        lines.append(f"Feedback time s: {result['feedback_time_s']:.8g}")
        lines.append(f"pk initial: {_format_vector(result['pk_initial'])}")
        lines.append(f"pk estimated: {_format_vector(result['pk_est'])}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    static_results_dir = script_dir / "results" / "Static_data"
    parser = argparse.ArgumentParser(
        description="Estimate SSM reduced states pk from StaticVerification measurements and cable-length targets.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=static_results_dir / "static_verification_max_displacement_measurements.npy",
        help="Input npy generated by plot_static_verification.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / "static_verification_pk_estimates_v2.npy",
        help="Output npy for estimated pk data.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=script_dir / "results" / "static_verification_pk_estimates_v2_log.txt",
        help="Output txt log for estimated pk and residual data.",
    )
    parser.add_argument(
        "--cable-lengths",
        type=Path,
        default=script_dir / "cable_lengths.xlsx",
        help="Excel file containing one cable-angle target row per Static_data sample.",
    )
    parser.add_argument(
        "--cable-weight",
        "--cw",
        type=float,
        default=0.2,
        help="Least-squares weight applied to cable-length residuals. Default: %(default)s.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=20,
        help="Maximum least-squares function evaluations.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use the previous estimated pk as the next initial guess.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = _load_npy_object(args.input)
    samples = loaded if isinstance(loaded, list) else [loaded]
    cable_targets = load_cable_targets(args.cable_lengths, samples)
    results = estimate_pk_for_samples(
        samples,
        cable_targets,
        sequential=args.sequential,
        cable_weight=args.cable_weight,
        max_nfev=args.max_nfev,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, results, allow_pickle=True)
    write_estimation_log(args.log, results)
    for result in results:
        print(
            f"{result['source']}: accepted={result['feedback_accepted']}, "
            f"measurement_residual={result['feedback_residual_norm']}, "
            f"cable_residual={result['cable_residual_norm']}, "
            f"saved pk shape={result['pk_est'].shape}"
        )
    print(f"Saved pk estimates to {args.output}")
    print(f"Saved pk estimation log to {args.log}")


if __name__ == "__main__":
    main()
