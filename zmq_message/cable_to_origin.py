from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zmq_message.zmq_control import DEFAULT_ENDPOINT, MOTOR_COUNT, close_publisher, create_publisher, send_motor_command


RAMP_DURATION_S = 5.0
RAMP_PERIOD_S = 0.075
PUBLISHER_WARMUP_S = 0.25
SWAP_MOTOR_PAIRS = False


def _format_vector(values: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def send_origin_command(
    socket,
    *,
    start_control: np.ndarray | None,
    duration_s: float,
    period_s: float,
    swap_motor_pairs: bool,
) -> dict:
    origin = np.zeros(MOTOR_COUNT, dtype=float)
    if start_control is None or duration_s <= 0:
        message = send_motor_command(
            socket,
            origin.tolist(),
            solver_order=False,
            swap_motor_pairs=swap_motor_pairs,
        )
        print(f"Sent origin command: {_format_vector(origin)}")
        return message

    if period_s <= 0:
        raise ValueError("--ramp-period-s must be positive")

    start_control = np.asarray(start_control, dtype=float).reshape(MOTOR_COUNT)
    step_count = max(1, int(np.ceil(duration_s / period_s)))
    final_message = None
    ramp_start = time.perf_counter()
    for step_index in range(step_count + 1):
        elapsed = min(step_index * period_s, duration_s)
        scale = float(np.cos(0.5 * np.pi * elapsed / duration_s))
        command = start_control * scale
        final_message = send_motor_command(
            socket,
            command.tolist(),
            solver_order=False,
            swap_motor_pairs=swap_motor_pairs,
        )
        print(
            f"Ramp {step_index:03d}/{step_count:03d} | "
            f"scale={scale:.4f} | command={_format_vector(command)}"
        )
        if step_index < step_count:
            next_elapsed = min((step_index + 1) * period_s, duration_s)
            sleep_s = ramp_start + next_elapsed - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    return final_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send all six cable/motor inputs back to zero.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="ZMQ motor publisher endpoint.")
    parser.add_argument(
        "--from-control",
        type=float,
        nargs=MOTOR_COUNT,
        metavar=("C1", "C2", "C3", "C4", "C5", "C6"),
        help="Optional current six-cable command. If provided, ramp smoothly from this command to zero.",
    )
    parser.add_argument("--ramp-s", type=float, default=RAMP_DURATION_S, help="Seconds for ramping to zero.")
    parser.add_argument(
        "--ramp-period-s",
        type=float,
        default=RAMP_PERIOD_S,
        help="Command update period during the ramp.",
    )
    parser.add_argument(
        "--publisher-warmup-s",
        type=float,
        default=PUBLISHER_WARMUP_S,
        help="Seconds to wait after binding before publishing the first command.",
    )
    parser.add_argument(
        "--swap-motor-pairs",
        action="store_true",
        default=SWAP_MOTOR_PAIRS,
        help="Swap motor channels 2/3 and 5/6 before sending.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = None
    socket = None
    try:
        context, socket = create_publisher(args.endpoint)
        print(f"ZeroMQ publisher started at {args.endpoint}")
        if args.publisher_warmup_s > 0:
            time.sleep(args.publisher_warmup_s)

        start_control = None if args.from_control is None else np.asarray(args.from_control, dtype=float)
        message = send_origin_command(
            socket,
            start_control=start_control,
            duration_s=float(args.ramp_s),
            period_s=float(args.ramp_period_s),
            swap_motor_pairs=bool(args.swap_motor_pairs),
        )
        motor_control_sent = np.asarray(message["param"]["posiSp"], dtype=float)
        print(f"Motor command sent after channel mapping: {_format_vector(motor_control_sent)}")
    finally:
        if socket is not None and context is not None:
            close_publisher(context, socket)


if __name__ == "__main__":
    main()
