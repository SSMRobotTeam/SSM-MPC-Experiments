# 输入一个电机控制，驱动机械臂，并测量
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measure.motion_capture import MotionReference, capture_motion, initialize_motion_reference
from utils.sdk_utils import load_lumo_sdk
from zmq_message.zmq_control import close_publisher, create_publisher, send_motor_command


IP_ADDRESS = "127.0.0.1"
MOTOR_ENDPOINT = "tcp://localhost:5555"
END_MARKER_ID = 100023
MID_RIGID_ID = 1
CHECKPOINT_RIGID_ID = 2
BASE_RIGID_ID = 0
MOTOR_COUNT = 6
TARGET_NAMES = ("endeff", "mideff", "checkpoint")

MOTOR_FACTOR = 1.225

# Input control values must be lengths and must use the original, unswapped order.
# Control_Inputs = [3.0080, 14.5448, 9.7830, 3.0002, 7.1149, -1.4481]
# Control_Inputs = [13.2430,	4.0946,	21.5010,	0.4657,	-2.0469,	9.3150]
# Control_Inputs = [12.6917,	17.9482,	1.3424,	0.5026,	4.7767,	-2.2942]
# Control_Inputs = [2.1763,	13.2501,	21.7237,	5.0435,	-2.2547,	0.7528]
# Control_Inputs = [7.9951,	9.1742,	2.3293,	0.1952,	9.4604,	-0.3789]
Control_Inputs = [6.3081,	2.0253,	6.4260,	-0.2197,	0.0698,	8.4275]

SWAP_MOTOR_PAIRS = False
RAMP_DURATION_S = 5.0
RAMP_PERIOD_S = 0.075

# # 输入的控制量需要是长度。
# Control_Inputs = [3.0080,	14.5448,	9.7830,	3.0002,	7.1149,	-1.4481]


def _format_vector(values: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def connect_sdk(ip_address: str):
    sdk_module = load_lumo_sdk(REPO_ROOT)
    sdk = sdk_module
    sdk.Init()
    if not sdk.Connect(ip_address):
        raise ConnectionError(f"Failed to connect LuMo SDK at {ip_address}")
    print(f"LuMo SDK connected successfully at {ip_address}")
    return sdk


def send_sine_ramp_command(
    socket,
    target_control: np.ndarray,
    *,
    duration_s: float,
    period_s: float,
) -> dict:
    target_control = np.asarray(target_control, dtype=float).reshape(MOTOR_COUNT)
    if duration_s <= 0:
        return send_motor_command(
            socket,
            target_control.tolist(),
            solver_order=False,
            swap_motor_pairs=SWAP_MOTOR_PAIRS,
        )
    if period_s <= 0:
        raise ValueError("--ramp-period-s must be positive")

    step_count = max(1, int(np.ceil(duration_s / period_s)))
    final_message = None
    ramp_start = time.perf_counter()
    for step_index in range(step_count + 1):
        elapsed = min(step_index * period_s, duration_s)
        scale = float(np.sin(0.5 * np.pi * elapsed / duration_s))
        command = target_control * scale
        final_message = send_motor_command(
            socket,
            command.tolist(),
            solver_order=False,
            swap_motor_pairs=SWAP_MOTOR_PAIRS,
        )
        print(
            f"Ramp {step_index:03d}/{step_count:03d} | "
            f"scale={scale:.4f} | command={_format_vector(command, 6)}"
        )
        if step_index < step_count:
            next_elapsed = min((step_index + 1) * period_s, duration_s)
            target_time = ramp_start + next_elapsed
            sleep_s = target_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    return final_message


def _reference_positions(reference: MotionReference) -> dict[str, np.ndarray]:
    return {
        "endeff": np.asarray(reference.end_effector, dtype=float).reshape(3),
        "mideff": np.asarray(reference.mid_effector, dtype=float).reshape(3),
        "checkpoint": np.asarray(reference.checkpoint, dtype=float).reshape(3),
    }


def _capture_once(
    sdk,
    reference: MotionReference,
    *,
    mid_rigid_id: int,
    checkpoint_rigid_id: int,
    end_marker_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    disp_end, disp_mid, disp_checkpoint = capture_motion(
        sdk,
        reference,
        mid_rigid_id=mid_rigid_id,
        checkpoint_rigid_id=checkpoint_rigid_id,
        end_marker_id=end_marker_id,
    )
    displacements = {
        "endeff": disp_end,
        "mideff": disp_mid,
        "checkpoint": disp_checkpoint,
    }
    initial_positions = _reference_positions(reference)
    positions = {name: initial_positions[name] + displacements[name] for name in TARGET_NAMES}
    return positions, displacements


def measure_stable_motion(
    sdk,
    reference: MotionReference,
    *,
    sample_count: int,
    sample_period_s: float,
    mid_rigid_id: int,
    checkpoint_rigid_id: int,
    end_marker_id: int,
) -> dict[str, dict[str, np.ndarray]]:
    position_samples = {name: [] for name in TARGET_NAMES}
    displacement_samples = {name: [] for name in TARGET_NAMES}

    for _ in range(sample_count):
        positions, displacements = _capture_once(
            sdk,
            reference,
            mid_rigid_id=mid_rigid_id,
            checkpoint_rigid_id=checkpoint_rigid_id,
            end_marker_id=end_marker_id,
        )
        for name in TARGET_NAMES:
            position_samples[name].append(positions[name])
            displacement_samples[name].append(displacements[name])
        if sample_period_s > 0:
            time.sleep(sample_period_s)

    results = {}
    for name in TARGET_NAMES:
        position_arr = np.asarray(position_samples[name], dtype=float)
        displacement_arr = np.asarray(displacement_samples[name], dtype=float)
        results[name] = {
            "position_ref_mm_mean": np.mean(position_arr, axis=0),
            "position_ref_mm_std": np.std(position_arr, axis=0),
            "displacement_ref_mm_mean": np.mean(displacement_arr, axis=0),
            "displacement_ref_mm_std": np.std(displacement_arr, axis=0),
        }
    return results


def write_csv(
    output_path: Path,
    raw_control: np.ndarray,
    control: np.ndarray,
    motor_control_sent: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "target",
        "raw_control_1",
        "raw_control_2",
        "raw_control_3",
        "raw_control_4",
        "raw_control_5",
        "raw_control_6",
        "control_1",
        "control_2",
        "control_3",
        "control_4",
        "control_5",
        "control_6",
        "motor_sent_1",
        "motor_sent_2",
        "motor_sent_3",
        "motor_sent_4",
        "motor_sent_5",
        "motor_sent_6",
        "position_x_ref_mm",
        "position_y_ref_mm",
        "position_z_ref_mm",
        "position_std_x_mm",
        "position_std_y_mm",
        "position_std_z_mm",
        "displacement_x_ref_mm",
        "displacement_y_ref_mm",
        "displacement_z_ref_mm",
        "displacement_std_x_mm",
        "displacement_std_y_mm",
        "displacement_std_z_mm",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for name in TARGET_NAMES:
            item = results[name]
            row = {"target": name}
            for idx, value in enumerate(raw_control):
                row[f"raw_control_{idx + 1}"] = float(value)
            for idx, value in enumerate(control):
                row[f"control_{idx + 1}"] = float(value)
            for idx, value in enumerate(motor_control_sent):
                row[f"motor_sent_{idx + 1}"] = float(value)
            for axis, value in zip(("x", "y", "z"), item["position_ref_mm_mean"]):
                row[f"position_{axis}_ref_mm"] = float(value)
            for axis, value in zip(("x", "y", "z"), item["position_ref_mm_std"]):
                row[f"position_std_{axis}_mm"] = float(value)
            for axis, value in zip(("x", "y", "z"), item["displacement_ref_mm_mean"]):
                row[f"displacement_{axis}_ref_mm"] = float(value)
            for axis, value in zip(("x", "y", "z"), item["displacement_ref_mm_std"]):
                row[f"displacement_std_{axis}_mm"] = float(value)
            writer.writerow(row)


def write_log(
    output_path: Path,
    raw_control: np.ndarray,
    control: np.ndarray,
    motor_control_sent: np.ndarray,
    ramp_duration_s: float,
    ramp_period_s: float,
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "StaticVerification Back_verify summary",
        f"Raw input control command: {_format_vector(raw_control, 6)}",
        f"Motor factor: {MOTOR_FACTOR:.6f}",
        f"Target control command after MOTOR_FACTOR: {_format_vector(control, 6)}",
        f"Swap motor pairs 2/3 and 5/6: {SWAP_MOTOR_PAIRS}",
        f"Motor command sent: {_format_vector(motor_control_sent, 6)}",
        f"Sine ramp duration s: {ramp_duration_s:.6f}",
        f"Sine ramp period s: {ramp_period_s:.6f}",
        "",
    ]
    for name in TARGET_NAMES:
        item = results[name]
        lines.append(f"{name}:")
        lines.append(f"  position_ref_mm_mean: {_format_vector(item['position_ref_mm_mean'], 6)}")
        lines.append(f"  position_ref_mm_std : {_format_vector(item['position_ref_mm_std'], 6)}")
        lines.append(f"  displacement_ref_mm_mean: {_format_vector(item['displacement_ref_mm_mean'], 6)}")
        lines.append(f"  displacement_ref_mm_std : {_format_vector(item['displacement_ref_mm_std'], 6)}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _timestamped_default_output_dir() -> Path:
    beijing_now = datetime.now(timezone(timedelta(hours=8)))
    return Path(__file__).resolve().parent / "results" / f"back_verify_{beijing_now:%Y%m%d_%H%M%S}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive motors with one 6-cable command and measure steady Reference-frame positions.",
    )
    parser.add_argument(
        "--control",
        type=float,
        nargs=6,
        metavar=("C1", "C2", "C3", "C4", "C5", "C6"),
        help="Six motor/cable length control values. Defaults to Control_Inputs at the top of this file.",
    )
    parser.add_argument("--endpoint", default=MOTOR_ENDPOINT, help="ZMQ motor publisher endpoint.")
    parser.add_argument("--ip", default=IP_ADDRESS, help="LuMo SDK server IP address.")
    parser.add_argument("--ramp-s", type=float, default=RAMP_DURATION_S, help="Seconds for sine ramp to the target command.")
    parser.add_argument(
        "--ramp-period-s",
        type=float,
        default=RAMP_PERIOD_S,
        help="Command update period during the sine ramp.",
    )
    parser.add_argument("--settle-s", type=float, default=5.0, help="Seconds to wait after sending control before sampling.")
    parser.add_argument("--samples", type=int, default=30, help="Number of motion-capture samples after settling.")
    parser.add_argument("--sample-period-s", type=float, default=0.05, help="Delay between motion-capture samples.")
    parser.add_argument("--base-y-offset", type=float, default=-12.0, help="Base Y offset passed to initialize_motion_reference.")
    parser.add_argument("--base-rigid-id", type=int, default=BASE_RIGID_ID)
    parser.add_argument("--mid-rigid-id", type=int, default=MID_RIGID_ID)
    parser.add_argument("--checkpoint-rigid-id", type=int, default=CHECKPOINT_RIGID_ID)
    parser.add_argument("--end-marker-id", type=int, default=END_MARKER_ID)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for npy/csv/log outputs.")
    parser.add_argument(
        "--return-zero",
        action="store_true",
        help="Send a zero command after measurement before closing the publisher.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    control_input = Control_Inputs if args.control is None else args.control
    raw_control = np.asarray(control_input, dtype=float).reshape(MOTOR_COUNT)
    control = raw_control * MOTOR_FACTOR
    output_dir = args.output_dir or _timestamped_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    sdk = None
    context = None
    socket = None
    try:
        sdk = connect_sdk(args.ip)
        reference = initialize_motion_reference(
            sdk,
            base_rigid_id=args.base_rigid_id,
            mid_rigid_id=args.mid_rigid_id,
            checkpoint_rigid_id=args.checkpoint_rigid_id,
            end_marker_id=args.end_marker_id,
            base_y_offset=args.base_y_offset,
        )
        print("Initial Reference positions (mm)")
        for name, value in _reference_positions(reference).items():
            print(f"  {name:10s}: {_format_vector(value, 6)}")

        context, socket = create_publisher(args.endpoint)
        time.sleep(0.25)
        print(f"Raw input control command: {_format_vector(raw_control, 6)}")
        print(f"MOTOR_FACTOR: {MOTOR_FACTOR:.6f}")
        print(f"Target command after MOTOR_FACTOR: {_format_vector(control, 6)}")
        print(
            f"Ramping to target command over {args.ramp_s:.3f}s "
            f"with update period {args.ramp_period_s:.3f}s..."
        )
        message = send_sine_ramp_command(
            socket,
            control,
            duration_s=args.ramp_s,
            period_s=args.ramp_period_s,
        )
        motor_control_sent = np.asarray(message["param"]["posiSp"], dtype=float)
        print(f"Reached target control command: {_format_vector(control, 6)}")
        print(f"Motor command after channel mapping: {_format_vector(motor_control_sent, 6)}")
        print(f"Waiting {args.settle_s:.3f}s for steady state...")
        if args.settle_s > 0:
            time.sleep(args.settle_s)

        results = measure_stable_motion(
            sdk,
            reference,
            sample_count=args.samples,
            sample_period_s=args.sample_period_s,
            mid_rigid_id=args.mid_rigid_id,
            checkpoint_rigid_id=args.checkpoint_rigid_id,
            end_marker_id=args.end_marker_id,
        )

        np.save(
            output_dir / "back_verify_measurement.npy",
            {
                "raw_control": raw_control,
                "control": control,
                "motor_factor": MOTOR_FACTOR,
                "motor_control_sent": motor_control_sent,
                "swap_motor_pairs": SWAP_MOTOR_PAIRS,
                "ramp_duration_s": args.ramp_s,
                "ramp_period_s": args.ramp_period_s,
                "motor_message": message,
                "initial_reference_positions_mm": _reference_positions(reference),
                "steady_measurements": results,
            },
            allow_pickle=True,
        )
        write_csv(output_dir / "back_verify_measurement.csv", raw_control, control, motor_control_sent, results)
        write_log(
            output_dir / "back_verify_measurement_log.txt",
            raw_control,
            control,
            motor_control_sent,
            args.ramp_s,
            args.ramp_period_s,
            results,
        )

        print("\nSteady Reference-frame positions")
        for name in TARGET_NAMES:
            item = results[name]
            print(
                f"  {name:10s}: pos={_format_vector(item['position_ref_mm_mean'], 6)} "
                f"disp={_format_vector(item['displacement_ref_mm_mean'], 6)} "
                f"std={_format_vector(item['position_ref_mm_std'], 6)}"
            )
        print(f"\nSaved Back_verify outputs to {output_dir}")

        if args.return_zero:
            send_motor_command(
                socket,
                np.zeros(MOTOR_COUNT).tolist(),
                solver_order=False,
                swap_motor_pairs=SWAP_MOTOR_PAIRS,
            )
            print("Sent zero return command.")
    finally:
        if socket is not None and context is not None:
            close_publisher(context, socket)
        if sdk is not None:
            sdk.Close()


if __name__ == "__main__":
    main()
