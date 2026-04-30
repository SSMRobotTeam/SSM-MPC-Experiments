import json
from datetime import datetime
from typing import Iterable

import zmq


DEFAULT_TOPIC = "/controller/setActuatorDate"
DEFAULT_ENDPOINT = "tcp://localhost:5555"
MOTOR_COUNT = 6


def reorder_motor_positions(positions: Iterable[float]) -> list[float]:
    """Map solver tendon order to the motor-controller order used in this setup."""
    p = list(positions)
    if len(p) != MOTOR_COUNT:
        raise ValueError(f"Expected {MOTOR_COUNT} motor positions, got {len(p)}")
    return [-p[0], -p[2], -p[1], -p[3], -p[5], -p[4]]


def build_motor_message(positions: Iterable[float], solver_order: bool = False) -> dict:
    posi_sp = reorder_motor_positions(positions) if solver_order else list(positions)
    if len(posi_sp) != MOTOR_COUNT:
        raise ValueError(f"Expected {MOTOR_COUNT} motor positions, got {len(posi_sp)}")

    return {
        "command": "set_motors",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "motorsCount": MOTOR_COUNT,
        "isCtlModeUnified": 1,
        "param": {
            "ctlMode": [3] * MOTOR_COUNT,
            "posiSp": posi_sp,
            "velcSp": [0] * MOTOR_COUNT,
            "forceSp": [0] * MOTOR_COUNT,
        },
    }


def create_publisher(endpoint: str = DEFAULT_ENDPOINT):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(endpoint)
    return context, socket


def send_motor_command(
    socket,
    positions: Iterable[float],
    topic: str = DEFAULT_TOPIC,
    solver_order: bool = False,
) -> dict:
    message = build_motor_message(positions, solver_order=solver_order)
    socket.send_multipart([topic.encode("utf-8"), json.dumps(message).encode("utf-8")])
    print(f"Sent @ {message['timestamp']}")
    return message


def close_publisher(context, socket) -> None:
    socket.close()
    context.term()
