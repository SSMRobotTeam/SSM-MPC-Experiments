from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def minimum_jerk_scale(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def minimum_jerk_scale_derivative(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    return 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4


def plan_minimum_jerk_trajectory(
    target_displacement_m: np.ndarray,
    *,
    dt: float,
    duration_s: float,
    hold_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan a point-to-point end-effector displacement trajectory.

    The path is a straight line in task space with quintic minimum-jerk
    time scaling. It starts at zero displacement and ends at target_displacement_m
    with zero velocity and zero acceleration at both endpoints.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    if hold_s < 0.0:
        raise ValueError("hold_s must be non-negative.")

    target = np.asarray(target_displacement_m, dtype=float).reshape(3)
    sample_count = int(np.ceil((duration_s + hold_s) / dt)) + 1
    time_s = np.arange(sample_count, dtype=float) * dt
    tau = np.minimum(time_s / duration_s, 1.0)
    scale = minimum_jerk_scale(tau)
    scale_dot = minimum_jerk_scale_derivative(tau) / duration_s
    scale_dot[time_s > duration_s] = 0.0

    position_m = target.reshape(3, 1) * scale.reshape(1, -1)
    velocity_m_s = target.reshape(3, 1) * scale_dot.reshape(1, -1)
    return time_s, position_m, velocity_m_s


def _perpendicular_direction(path_direction: np.ndarray, preferred_axis: np.ndarray) -> np.ndarray:
    direction = preferred_axis - path_direction * float(preferred_axis @ path_direction)
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        return direction / norm

    for axis in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        direction = axis - path_direction * float(axis @ path_direction)
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            return direction / norm
    return np.array([0.0, 1.0, 0.0])


def obstacle_avoidance_waypoints(
    start_position_m: np.ndarray,
    target_position_m: np.ndarray,
    obstacle_positions_m: np.ndarray,
    clearance_m: float,
) -> np.ndarray:
    start = np.asarray(start_position_m, dtype=float).reshape(3)
    target = np.asarray(target_position_m, dtype=float).reshape(3)
    obstacles = np.asarray(obstacle_positions_m, dtype=float).reshape(-1, 3)
    path = target - start
    path_length = np.linalg.norm(path)
    if path_length <= 1e-12 or obstacles.size == 0:
        return np.empty((0, 3))

    path_direction = path / path_length
    waypoints = []
    for obstacle in obstacles:
        alpha = float(((obstacle - start) @ path) / (path_length**2))
        if alpha <= 0.0 or alpha >= 1.0:
            continue

        closest = start + alpha * path
        obstacle_to_path = closest - obstacle
        distance = np.linalg.norm(obstacle_to_path)
        if distance >= clearance_m:
            continue

        if distance > 1e-9:
            lateral_direction = obstacle_to_path / distance
        else:
            lateral_direction = _perpendicular_direction(path_direction, np.array([0.0, 1.0, 0.0]))
        waypoint = closest + lateral_direction * (clearance_m - distance + clearance_m)
        waypoint_alpha = float(((waypoint - start) @ path) / (path_length**2))
        waypoints.append((waypoint_alpha, waypoint))

    if not waypoints:
        return np.empty((0, 3))

    waypoints.sort(key=lambda item: item[0])
    return np.asarray([waypoint for _, waypoint in waypoints], dtype=float).reshape(-1, 3)


def plan_minimum_jerk_waypoint_trajectory(
    waypoints_m: np.ndarray,
    *,
    dt: float,
    duration_s: float,
    hold_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    waypoints = np.asarray(waypoints_m, dtype=float).reshape(-1, 3)
    if waypoints.shape[0] < 2:
        raise ValueError("At least start and target waypoints are required.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    if hold_s < 0.0:
        raise ValueError("hold_s must be non-negative.")

    segment_vectors = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 1e-12:
        return plan_minimum_jerk_trajectory(np.zeros(3), dt=dt, duration_s=duration_s, hold_s=hold_s)

    segment_durations = duration_s * segment_lengths / total_length
    min_segment_duration = max(dt, 0.05 * duration_s / max(1, segment_lengths.size))
    segment_durations = np.maximum(segment_durations, min_segment_duration)
    segment_durations *= duration_s / float(np.sum(segment_durations))
    segment_start_times = np.concatenate([[0.0], np.cumsum(segment_durations)])

    sample_count = int(np.ceil((duration_s + hold_s) / dt)) + 1
    time_s = np.arange(sample_count, dtype=float) * dt
    position_m = np.zeros((3, sample_count))
    velocity_m_s = np.zeros((3, sample_count))

    for sample_idx, sample_time in enumerate(time_s):
        if sample_time >= duration_s:
            position_m[:, sample_idx] = waypoints[-1]
            continue

        segment_idx = int(np.searchsorted(segment_start_times, sample_time, side="right") - 1)
        segment_idx = min(max(segment_idx, 0), segment_lengths.size - 1)
        local_time = sample_time - segment_start_times[segment_idx]
        segment_duration = segment_durations[segment_idx]
        tau = np.array([local_time / segment_duration])
        scale = minimum_jerk_scale(tau)[0]
        scale_dot = minimum_jerk_scale_derivative(tau)[0] / segment_duration
        segment = waypoints[segment_idx + 1] - waypoints[segment_idx]
        position_m[:, sample_idx] = waypoints[segment_idx] + scale * segment
        velocity_m_s[:, sample_idx] = scale_dot * segment

    return time_s, position_m, velocity_m_s


def plan_obstacle_avoidance_trajectory(
    target_displacement_m: np.ndarray,
    obstacle_positions_m: np.ndarray,
    *,
    start_position_m: np.ndarray,
    clearance_m: float,
    dt: float,
    duration_s: float,
    hold_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(start_position_m, dtype=float).reshape(3)
    target_position = start + np.asarray(target_displacement_m, dtype=float).reshape(3)
    detour_waypoints = obstacle_avoidance_waypoints(start, target_position, obstacle_positions_m, clearance_m)
    absolute_waypoints = np.vstack([start, detour_waypoints, target_position])
    time_s, absolute_position_m, velocity_m_s = plan_minimum_jerk_waypoint_trajectory(
        absolute_waypoints,
        dt=dt,
        duration_s=duration_s,
        hold_s=hold_s,
    )
    displacement_m = absolute_position_m - start.reshape(3, 1)
    return time_s, displacement_m, velocity_m_s, absolute_waypoints


def trajectory_obstacle_clearance(position_m: np.ndarray, obstacle_positions_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(position_m, dtype=float)
    if positions.shape[0] != 3:
        positions = positions.T
    obstacles = np.asarray(obstacle_positions_m, dtype=float).reshape(-1, 3)
    if obstacles.size == 0:
        return np.empty((0,))
    deltas = positions.T[:, None, :] - obstacles[None, :, :]
    return np.min(np.linalg.norm(deltas, axis=2), axis=0)


def write_trajectory_csv(
    output_path: Path,
    time_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "time_s",
        "disp_x_m",
        "disp_y_m",
        "disp_z_m",
        "vel_x_m_s",
        "vel_y_m_s",
        "vel_z_m_s",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for idx, time_value in enumerate(np.asarray(time_s, dtype=float).reshape(-1)):
            writer.writerow(
                {
                    "time_s": float(time_value),
                    "disp_x_m": float(position_m[0, idx]),
                    "disp_y_m": float(position_m[1, idx]),
                    "disp_z_m": float(position_m[2, idx]),
                    "vel_x_m_s": float(velocity_m_s[0, idx]),
                    "vel_y_m_s": float(velocity_m_s[1, idx]),
                    "vel_z_m_s": float(velocity_m_s[2, idx]),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a minimum-jerk point-to-point end-effector trajectory.")
    parser.add_argument(
        "--target-mm",
        type=float,
        nargs=3,
        required=True,
        metavar=("DX", "DY", "DZ"),
        help="Target end-effector displacement in millimeters.",
    )
    parser.add_argument("--dt", type=float, default=0.075, help="Sampling period in seconds.")
    parser.add_argument("--duration", type=float, default=8.0, help="Motion duration in seconds.")
    parser.add_argument("--hold", type=float, default=2.0, help="Extra hold time at the target in seconds.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PlotFig") / "planned_minimum_jerk_trajectory.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("PlotFig") / "planned_minimum_jerk_trajectory.npz",
        help="NPZ output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_m = np.asarray(args.target_mm, dtype=float).reshape(3) / 1000.0
    time_s, position_m, velocity_m_s = plan_minimum_jerk_trajectory(
        target_m,
        dt=args.dt,
        duration_s=args.duration,
        hold_s=args.hold,
    )
    write_trajectory_csv(args.output, time_s, position_m, velocity_m_s)
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.npz,
        t=time_s,
        zd=position_m,
        velocity_m_s=velocity_m_s,
        target_displacement_m=target_m,
    )
    print("Planned minimum-jerk trajectory.")
    print(f"Target displacement mm: {np.asarray(args.target_mm, dtype=float).reshape(3)}")
    print(f"Duration s: {args.duration}, hold s: {args.hold}, dt s: {args.dt}")
    print(f"Saved CSV to {args.output}")
    print(f"Saved NPZ to {args.npz}")


if __name__ == "__main__":
    main()
