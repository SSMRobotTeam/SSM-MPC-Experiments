import math
import time

import numpy as np

from zmq_message.zmq_control import close_publisher, create_publisher, send_motor_command


def send_zmq_messages(period_seconds: float = 0.1) -> None:
    context, socket = create_publisher("tcp://localhost:5555")
    print("ZeroMQ publisher started on /controller/setActuatorDate")

    try:
        start_time = time.time()
        while True:
            t = time.time() - start_time
            positions = [4 * math.sin(2 * np.pi * t / 40), 0, 0, 0, 0, 0]
            send_motor_command(socket, positions)
            time.sleep(period_seconds)
    except KeyboardInterrupt:
        print("\nPublisher stopped by user.")
    finally:
        close_publisher(context, socket)


if __name__ == "__main__":
    send_zmq_messages()
