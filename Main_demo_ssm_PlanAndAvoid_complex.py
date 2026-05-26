#!/usr/bin/python
import argparse
import sys
import time

import numpy as np
import scipy.io as sciIO

from controller.cable_constraints import cable_length_double_part
from controller.nmpc import nmpc_multi_shooting_zoh_avoid_complex
from controller.ssm_controller import SsmExperimentConfig, SsmMpcController
from controller.trajectory_planner import plan_obstacle_avoidance_trajectory, trajectory_obstacle_clearance
from dynamics.SSM.matlab_data import get_field, load_rom_data
from Main_demo_ssm_avoid_complex import (
    CONTROL_PERIOD_SECONDS,
    CONTROL_SMOOTH_WEIGHT,
    FEEDBACK_ENABLED,
    LOG_DIR,
    MODEL_INFO_DIR,
    MOTOR_ENDPOINT,
    MOTOR_FACTOR,
    MPC_SOLVER_TYPE,
    OBSTACLE_RIGID_IDS,
    OBSTACLE_RADIUS,
    PLOT_DIR,
    POS_END,
    QP_BACKEND,
    Q_WEIGHT,
    R_WEIGHT,
    SAFE_DISTANCE_FACTOR,
    SAVE_FEEDBACK_EACH_STEP,
    SPIRAL_TRAJ_DIR,
    USE_FAST_SINGLE_STEP_QP,
    BufferedTeeLogger,
    build_initial_feedback_state,
    build_ssm_maps,
    cable_displacement_size,
    close_publisher,
    connect_sdk,
    create_publisher,
    format_motor_positions,
    horizon,
    initialize_avoidance_reference,
    print_timing_summary,
    run_step,
)


PLAN_OUTPUT_SUFFIX = "_plan_avoid_complex"
PLAN_RESULT_DIR = PLOT_DIR / "Plan_And_Avoid"

# Global inputs for this demo. Edit these values before running the script.
TARGET_DISPLACEMENT_MM = np.array([120.0, -40.0, -60.0])
RUN_STEPS = 320
PLAN_DURATION_S = 20.0
TARGET_HOLD_S = 2.0
STOP_ON_TARGET = True
PLANNER_CLEARANCE_FACTOR = 3 # 2.5

TARGET_TOLERANCE_MM = 2.0


def build_plan_avoid_config(rom_data, avoid_ref, td: np.ndarray, zd: np.ndarray) -> SsmExperimentConfig:
    node_list = rom_data["Node_List"]
    u_lb = np.full(6 * horizon, -50)
    u_ub = np.full(6 * horizon, 50)
    dist0 = cable_length_double_part(np.zeros(cable_displacement_size(node_list)), node_list)

    return SsmExperimentConfig(
        dt=CONTROL_PERIOD_SECONDS,
        tf=float(td[-1]),
        td=td,
        zd=zd,
        Np=horizon,
        Q=Q_WEIGHT,
        R=R_WEIGHT,
        u_lb=u_lb,
        u_ub=u_ub,
        Node_List=node_list,
        dist0=dist0,
        opt_W=36,
        solver_type=MPC_SOLVER_TYPE,
        qp_backend=QP_BACKEND,
        use_fast_single_step=USE_FAST_SINGLE_STEP_QP,
        control_smooth_weight=CONTROL_SMOOTH_WEIGHT,
        nmpc_solver=nmpc_multi_shooting_zoh_avoid_complex,
        pos_end=POS_END,
        pos_obstacle=avoid_ref.obstacle_positions,
        avoid_node_positions=avoid_ref.obstacle_node_positions,
        d_safe=SAFE_DISTANCE_FACTOR * OBSTACLE_RADIUS,
        motor_safety_limit=40.0,
    )


def target_error_mm(controller: SsmMpcController, target_displacement_m: np.ndarray) -> float:
    endeff_displacement_mm = np.asarray(controller.yk_ssm[:3], dtype=float).reshape(3)
    target_mm = np.asarray(target_displacement_m, dtype=float).reshape(3) * 1000.0
    return float(np.linalg.norm(endeff_displacement_mm - target_mm))


def main(
    target_displacement_mm: np.ndarray = TARGET_DISPLACEMENT_MM,
    run_steps: int = RUN_STEPS,
    stop_on_target: bool = STOP_ON_TARGET,
    plan_duration_s: float = PLAN_DURATION_S,
    target_hold_s: float = TARGET_HOLD_S,
) -> None:
    original_stdout = sys.stdout
    logger = BufferedTeeLogger(original_stdout)
    sys.stdout = logger

    sdk = None
    context = None
    socket = None
    timing_records = []
    max_mpc_time = 0.0
    completed_steps = 0
    target_displacement_m = np.asarray(target_displacement_mm, dtype=float).reshape(3) / 1000.0
    try:
        print("Using SSM point-to-point planning with complex obstacle avoidance.")
        print(f"Target end-effector displacement (mm): {target_displacement_mm}")
        rom_data = load_rom_data(MODEL_INFO_DIR, traj_dir=SPIRAL_TRAJ_DIR)
        x_num = np.asarray(get_field(rom_data["auData"], "lamd")).reshape(-1).size

        sdk = connect_sdk()
        avoid_ref = initialize_avoidance_reference(sdk, rom_data, use_cached_obstacles=False)

        planner_clearance = PLANNER_CLEARANCE_FACTOR * OBSTACLE_RADIUS
        td_plan, zd_plan, velocity_plan, planner_waypoints = plan_obstacle_avoidance_trajectory(
            target_displacement_m,
            avoid_ref.obstacle_positions,
            start_position_m=POS_END,
            clearance_m=planner_clearance,
            dt=CONTROL_PERIOD_SECONDS,
            duration_s=plan_duration_s,
            hold_s=target_hold_s,
        )
        absolute_plan = zd_plan + POS_END.reshape(3, 1)
        obstacle_clearances = trajectory_obstacle_clearance(absolute_plan, avoid_ref.obstacle_positions)
        print(
            "Planner: obstacle-aware waypoint minimum-jerk trajectory, "
            f"duration={plan_duration_s:.3f}s, hold={target_hold_s:.3f}s, samples={td_plan.size}."
        )
        print(f"Planner clearance target (m): {planner_clearance:.6f}")
        print(f"Planner absolute waypoints (m): {planner_waypoints}")
        if obstacle_clearances.size:
            print(f"Planned minimum obstacle clearances (m): {obstacle_clearances}")

        config = build_plan_avoid_config(rom_data, avoid_ref, td_plan, zd_plan)
        w0_for_out, w0_for_cables, w0_for_feedback, w0_for_avoid_nodes = build_ssm_maps(
            rom_data,
            avoid_ref,
        )

        print(f"Obstacle positions in model frame (m): {config.pos_obstacle}")
        for obstacle_idx, (node_index, node_position) in enumerate(
            zip(avoid_ref.obstacle_node_indices_1based, avoid_ref.obstacle_node_positions),
            start=1,
        ):
            print(f"  obstacle {obstacle_idx}: nearest node {node_index}, node position (m) {node_position}")
        print(f"Safe distance (m): {config.d_safe:.6f}")

        controller = SsmMpcController(
            config=config,
            au_data=rom_data["auData"],
            w0_for_out=w0_for_out,
            w0_for_cables=w0_for_cables,
            w0_for_feedback=w0_for_feedback,
            w0_for_avoid_nodes=w0_for_avoid_nodes,
            feedback_state=build_initial_feedback_state(x_num, 6, config.Np),
            factor=MOTOR_FACTOR,
            feedback_enabled=FEEDBACK_ENABLED,
            save_feedback_each_step=SAVE_FEEDBACK_EACH_STEP,
        )

        context, socket = create_publisher(MOTOR_ENDPOINT)
        step_count = min(int(run_steps), config.zd.shape[1])
        y_expm = np.zeros((step_count, 9))
        y_ssm = np.zeros((step_count, 6))
        pk_save = np.zeros((step_count, x_num))
        u_save = np.zeros((step_count, 6))
        target_error_save = np.zeros(step_count)

        for step_index in range(step_count):
            start = time.perf_counter()
            motor_positions, measure_time, mpc_time, feedback_time = run_step(
                sdk,
                socket,
                controller,
                avoid_ref,
                step_index,
                y_expm,
                y_ssm,
                u_save,
                feedback=FEEDBACK_ENABLED,
            )
            y_ssm[step_index, :6] = controller.yk_ssm
            pk_save[step_index, :] = controller.feedback_state.pk
            target_error_save[step_index] = target_error_mm(controller, target_displacement_m)
            completed_steps = step_index + 1
            total_time = time.perf_counter() - start
            max_mpc_time = max(max_mpc_time, mpc_time)
            timing_records.append(
                {"total": total_time, "measure": measure_time, "feedback": feedback_time, "mpc": mpc_time}
            )
            print(
                f"Step {step_index + 1:04d}/{step_count:04d} | "
                f"target error {target_error_save[step_index]:8.3f} mm | "
                f"total {total_time:8.4f}s | "
                f"measure {measure_time:8.4f}s | "
                f"feedback {feedback_time:8.4f}s | "
                f"mpc {mpc_time:8.4f}s | "
                f"mpc max {max_mpc_time:8.4f}s | "
                f"motor {format_motor_positions(motor_positions)}"
            )
            if stop_on_target and target_error_save[step_index] <= TARGET_TOLERANCE_MM:
                print(f"Target reached within {TARGET_TOLERANCE_MM:.3f} mm tolerance.")
                break

        PLAN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        slice_steps = slice(0, completed_steps)
        t_save = config.td[:completed_steps]
        np.save(PLAN_RESULT_DIR / f"y_exprm_ssm{PLAN_OUTPUT_SUFFIX}", y_expm[slice_steps])
        np.save(PLAN_RESULT_DIR / f"y_ssm{PLAN_OUTPUT_SUFFIX}", y_ssm[slice_steps])
        np.save(PLAN_RESULT_DIR / f"pk_save_ssm{PLAN_OUTPUT_SUFFIX}", pk_save[slice_steps])
        np.save(PLAN_RESULT_DIR / f"ut_save_ssm{PLAN_OUTPUT_SUFFIX}", u_save[slice_steps])
        np.save(PLAN_RESULT_DIR / f"target_error_ssm{PLAN_OUTPUT_SUFFIX}", target_error_save[slice_steps])
        np.save(PLAN_RESULT_DIR / f"t_ssm{PLAN_OUTPUT_SUFFIX}", t_save)
        np.save(PLAN_RESULT_DIR / f"planned_zd_ssm{PLAN_OUTPUT_SUFFIX}", zd_plan[:, :completed_steps])
        np.save(PLAN_RESULT_DIR / f"obstacle_positions_ssm{PLAN_OUTPUT_SUFFIX}", config.pos_obstacle)
        np.save(PLAN_RESULT_DIR / f"planner_waypoints_ssm{PLAN_OUTPUT_SUFFIX}", planner_waypoints)
        sciIO.savemat(
            PLAN_RESULT_DIR / f"ssm_python_results{PLAN_OUTPUT_SUFFIX}.mat",
            {
                "y_exprm": y_expm[slice_steps],
                "y_ssm": y_ssm[slice_steps],
                "pk_save": pk_save[slice_steps],
                "ut_save": u_save[slice_steps],
                "target_error_mm": target_error_save[slice_steps],
                "t": t_save,
                "planned_zd": zd_plan[:, :completed_steps],
                "planned_velocity_m_s": velocity_plan[:, :completed_steps],
                "planner_waypoints_m": planner_waypoints,
                "planner_clearance_m": planner_clearance,
                "planned_obstacle_clearance_m": obstacle_clearances,
                "target_displacement_m": target_displacement_m,
                "target_displacement_mm": np.asarray(target_displacement_mm, dtype=float).reshape(3),
                "pos_end": POS_END,
                "pos_obstacle": config.pos_obstacle,
                "obstacle_rigid_ids": np.asarray(OBSTACLE_RIGID_IDS, dtype=int).reshape(-1),
                "obstacle_node_indices_1based": avoid_ref.obstacle_node_indices_1based,
                "obstacle_node_positions": avoid_ref.obstacle_node_positions,
                "obstacle_node_full_dofs_0based": avoid_ref.obstacle_node_full_dofs_0based,
                "obstacle_radius": OBSTACLE_RADIUS,
                "d_safe": config.d_safe,
            },
        )
    finally:
        print_timing_summary(timing_records)
        if socket is not None and context is not None:
            close_publisher(context, socket)
        if sdk is not None:
            if hasattr(sdk, "Disconnect"):
                sdk.Disconnect()
            sdk.Close()
        sys.stdout = original_stdout
        log_path = logger.save(LOG_DIR)
        print(f"Log saved to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSM point-to-point planning with complex obstacle avoidance.")
    parser.add_argument(
        "--target-mm",
        type=float,
        nargs=3,
        default=TARGET_DISPLACEMENT_MM,
        metavar=("DX", "DY", "DZ"),
        help="Optional override for TARGET_DISPLACEMENT_MM, in millimeters.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=RUN_STEPS,
        help="Optional override for RUN_STEPS. The actual count is also limited by the planned trajectory length.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=PLAN_DURATION_S,
        help="Optional override for PLAN_DURATION_S.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=TARGET_HOLD_S,
        help="Optional override for TARGET_HOLD_S.",
    )
    parser.add_argument(
        "--no-stop-on-target",
        action="store_true",
        default=not STOP_ON_TARGET,
        help="Override STOP_ON_TARGET=False and keep running until --steps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        target_displacement_mm=np.asarray(args.target_mm, dtype=float),
        run_steps=int(args.steps),
        stop_on_target=not args.no_stop_on_target,
        plan_duration_s=float(args.duration),
        target_hold_s=float(args.hold),
    )
