#!/usr/bin/python
from pathlib import Path

from utils.sdk_utils import load_lumo_sdk


IP_ADDRESS = "127.0.0.1"


def print_camera_list(sdk) -> None:
    if not sdk.ReceiveCameraData():
        return

    for camera_id, camera in sdk.GetCameraList().items():
        print(
            f"CameraId: {camera_id} | Ip: {camera.Ip} | Serial: {camera.Serial} | "
            f"Model: {camera.Model} | Exposure: {camera.Exposure} | "
            f"Gain: {camera.Gain} | FrameRate: {camera.FrameRate} | MTU: {camera.MTU}"
        )


def print_rigid_info(rigid_info) -> None:
    fields = [
        "RigidName",
        "X",
        "Y",
        "Z",
        "qx",
        "qy",
        "qz",
        "qw",
        "IsTrack",
        "QualityGrade",
        "fSpeed",
        "fXSpeed",
        "fYSpeed",
        "fZSpeed",
        "fAcceleratedSpeed",
        "fXAcceleratedSpeed",
        "fYAcceleratedSpeed",
        "fZAcceleratedSpeed",
        "fXEulerAngle",
        "fYEulerAngle",
        "fZEulerAngle",
        "fXPalstance",
        "fYPalstance",
        "fZPalstance",
        "AccfXPalstance",
        "AccfYPalstance",
        "AccfZPalstance",
    ]
    for rigid_id, rigid in rigid_info.items():
        values = " | ".join(f"{field}: {getattr(rigid, field)}" for field in fields)
        print(f"Rigid ID: {rigid_id} | {values}")


def print_marker_info(marker_info) -> None:
    for marker_id, marker in marker_info.items():
        print(
            f"MarkerID: {marker_id} | MarkerName: {marker.MarkerName} | "
            f"X: {marker.X} | Y: {marker.Y} | Z: {marker.Z}"
        )


def print_body_info(body_info) -> None:
    for body_id, body in body_info.items():
        print(f"BodyID: {body_id} | BodyName: {body.BodyName} | IsTrack: {body.IsTrack}")
        for joint in list(body.vecJointNodes):
            print(
                f"BodyID: {body_id} | JointID: {joint.iJointID} | "
                f"JointName: {joint.sJointName} | X: {joint.X} | Y: {joint.Y} | Z: {joint.Z} | "
                f"qx: {joint.qx} | qy: {joint.qy} | qz: {joint.qz} | qw: {joint.qw}"
            )


def print_marker_sets(marker_set) -> None:
    for set_name, marker_group in marker_set.items():
        print(f"MarkerSetName: {set_name} | Name: {marker_group.sName}")
        for marker in list(marker_group.vmarkers):
            print(
                f"MarkerSetName: {set_name} | MarkerID: {marker.MarkerID} | "
                f"MarkerName: {marker.MarkerName} | X: {marker.X} | Y: {marker.Y} | Z: {marker.Z}"
            )


def print_force_plate(force_plate) -> None:
    fields = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Lx", "Lz"]
    print("ForcePlate | " + " | ".join(f"{field}: {getattr(force_plate, field)}" for field in fields))


def print_frame_snapshot(sdk) -> None:
    sdk.ReceiveData(0)
    print(f"FrameId: {sdk.GetFramId()}")
    print(f"TimeStamp: {sdk.GetTimeStamp()}")
    print(f"CameraSyncTime: {sdk.GetCameraSyncTime()}")
    print(f"BroadcastTime: {sdk.GetBroadcastTime()}")

    print_rigid_info(sdk.GetRigid())
    print_marker_info(sdk.GetMarker())
    print_body_info(sdk.GetBodysPose())
    print_marker_sets(sdk.GetMarkerSet())
    print_force_plate(sdk.GetForcePlateData())


def main() -> None:
    sdk = load_lumo_sdk(Path(__file__).resolve().parent.parent)
    sdk.Init()

    try:
        if sdk.Connect(IP_ADDRESS):
            print("connect successfully")
        else:
            raise ConnectionError(f"Failed to connect LuMo SDK at {IP_ADDRESS}")

        print_camera_list(sdk)
        while True:
            print_frame_snapshot(sdk)
    finally:
        sdk.Disconnect()
        sdk.Close()


if __name__ == "__main__":
    main()
