from zmq_message.zmq_control import send_motor_command


def mess_sent(socket_host, posiMess=None):
    """Backward-compatible wrapper for the old motor command helper."""
    if posiMess is None:
        posiMess = [0, 0, 0, 0, 0, 0]
    return send_motor_command(socket_host, posiMess)


if __name__ == "__main__":
    from zmq_message.zmq_control import close_publisher, create_publisher

    context, socket = create_publisher("tcp://*:5552")
    try:
        print("Publisher started at tcp://*:5552")
        mess_sent(socket, [1, 1, 1, 1, 1, 1])
    finally:
        close_publisher(context, socket)
