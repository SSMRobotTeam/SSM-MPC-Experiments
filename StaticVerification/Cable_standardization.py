from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller.cable_constraints import cable_length_double_part
from dynamics.SSM.matlab_data import ROM_DIR, clone_struct_array, get_field, load_rom_data, subset_mapping_rows
from dynamics.SSM.ssm_dynamics import reduced_to_full
from zmq_message.zmq_control import reorder_motor_positions


SSM_FACTOR = 1.475
CABLE_NAMES = ("long_1", "long_2", "long_3", "short_1", "short_2", "short_3")


def _load_npy_object(path: Path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    return data.tolist()


def _as_list(value) -> list[dict]:
    return value if isinstance(value, list) else [value]


def _scaled_w0(rom_data: dict):
    w0 = clone_struct_array(rom_data["W_0"])
    if len(w0) > 1:
        w0[1].coeffs = np.asarray(get_field(w0[1], "coeffs")) * SSM_FACTOR
    return w0


def _cable_dofs_for_wp(node_list) -> np.ndarray:
    dofs = []
    for section_idx, last_node in [(0, 17), (1, 10)]:
        for cable_idx in range(3):
            for node_idx in range(1, last_node):
                dofs.extend(np.asarray(node_list[section_idx][cable_idx, node_idx]).reshape(-1)[3:6])
    return np.asarray(dofs, dtype=int)


def _cable_displacement_size(node_list) -> int:
    long_count = 3 * (node_list[0].shape[1] - 1) * 3
    short_count = 3 * (node_list[1].shape[1] - 1) * 3
    return long_count + short_count


def _build_w0_for_cables(rom_data: dict):
    return subset_mapping_rows(_scaled_w0(rom_data), _cable_dofs_for_wp(rom_data["Node_List"]))


def calculate_cable_control_info(estimates: list[dict], rom_data: dict) -> list[dict]:
    node_list = rom_data["Node_List"]
    w0_for_cables = _build_w0_for_cables(rom_data)
    baseline_length_m = cable_length_double_part(np.zeros(_cable_displacement_size(node_list)), node_list)

    results = []
    for estimate in estimates:
        pk_est = np.asarray(estimate["pk_est"], dtype=float).reshape(-1)
        cable_displacement = reduced_to_full(pk_est, w0_for_cables, None, 1).reshape(-1)
        cable_length_m = cable_length_double_part(cable_displacement, node_list)
        length_delta_m = cable_length_m - baseline_length_m
        length_delta_mm = length_delta_m * 1000.0

        results.append(
            {
                "source": estimate["source"],
                "feedback_accepted": bool(estimate.get("feedback_accepted", False)),
                "feedback_residual_norm": estimate.get("feedback_residual_norm"),
                "pk_est": pk_est,
                "cable_displacement": cable_displacement,
                "baseline_length_m": baseline_length_m,
                "cable_length_m": cable_length_m,
                "length_delta_m": length_delta_m,
                "length_delta_mm": length_delta_mm,
                "motor_order_length_delta_mm": np.asarray(reorder_motor_positions(length_delta_mm), dtype=float),
            }
        )
    return results


def _format_vector(values: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def write_log(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "StaticVerification cable standardization summary",
        "Cable order: " + ", ".join(CABLE_NAMES),
        "length_delta = estimated cable length - baseline cable length",
        "",
    ]
    for result in results:
        lines.append(f"Source: {result['source']}")
        lines.append(f"Feedback accepted: {result['feedback_accepted']}")
        lines.append(f"Feedback residual norm: {result['feedback_residual_norm']}")
        lines.append(f"Baseline length m: {_format_vector(result['baseline_length_m'], 8)}")
        lines.append(f"Estimated length m: {_format_vector(result['cable_length_m'], 8)}")
        lines.append(f"Length delta mm: {_format_vector(result['length_delta_mm'], 6)}")
        lines.append(f"Motor-order length delta mm: {_format_vector(result['motor_order_length_delta_mm'], 6)}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["source", "feedback_accepted", "feedback_residual_norm"]
    for prefix in ("baseline_length_m", "cable_length_m", "length_delta_m", "length_delta_mm"):
        headers.extend([f"{prefix}_{name}" for name in CABLE_NAMES])
    headers.extend([f"motor_order_length_delta_mm_{idx + 1}" for idx in range(6)])

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for result in results:
            row = {
                "source": result["source"],
                "feedback_accepted": result["feedback_accepted"],
                "feedback_residual_norm": result["feedback_residual_norm"],
            }
            for prefix in ("baseline_length_m", "cable_length_m", "length_delta_m", "length_delta_mm"):
                for name, value in zip(CABLE_NAMES, result[prefix]):
                    row[f"{prefix}_{name}"] = float(value)
            for idx, value in enumerate(result["motor_order_length_delta_mm"]):
                row[f"motor_order_length_delta_mm_{idx + 1}"] = float(value)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compute static cable-length control information from estimated SSM pk values.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "results" / "static_verification_pk_estimates.npy",
        help="Input npy generated by estimate_static_pk.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / "static_verification_cable_standardization.npy",
        help="Output npy containing cable length and control information.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=script_dir / "results" / "static_verification_cable_standardization.csv",
        help="Output CSV summary.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=script_dir / "results" / "static_verification_cable_standardization_log.txt",
        help="Output text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimates = _as_list(_load_npy_object(args.input))
    rom_data = load_rom_data(ROM_DIR)
    results = calculate_cable_control_info(estimates, rom_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, results, allow_pickle=True)
    write_csv(args.csv, results)
    write_log(args.log, results)

    for result in results:
        print(f"{result['source']}: length_delta_mm={_format_vector(result['length_delta_mm'], 4)}")
    print(f"Saved cable standardization data to {args.output}")
    print(f"Saved CSV summary to {args.csv}")
    print(f"Saved log summary to {args.log}")


if __name__ == "__main__":
    main()
