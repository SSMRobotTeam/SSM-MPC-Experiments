#!/usr/bin/python
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import scipy.io as sciIO

from dynamics.pcs_length_static_forward import (
    PCSLengthStaticOptions,
    solve_pcs_length_static_forward_point,
)
from dynamics.pcs_tension_static_optimizer import (
    PCSBeamParameters,
    PCSSingleSegmentModel,
    PCSTwoSegmentsModel,
)
# Temporarily disabled motion capture while tuning PCS static MPC.
# from measure.motion_capture import capture_motion, initialize_motion_reference
from utils.sdk_utils import load_lumo_sdk
from utils.timing_utils import fixed_cpu_time
from zmq_message.zmq_control import clamp_motor_positions, close_publisher, create_publisher, send_motor_command


BASE_DIR = Path(__file__).resolve().parent
TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/spiral"
# TRAJ_DIR = BASE_DIR / "dynamics/trajinfom/eight"
PLOT_DIR = BASE_DIR / "PlotFig"
RESULT_DIR = PLOT_DIR / "pcs_static_results"
MOTOR_ENDPOINT = "tcp://localhost:5555"

IP_ADDRESS = "127.0.0.1"
CONTROL_PERIOD_SECONDS = 0.075
MPC_HORIZON = 1

# The static PCS model uses the model x-axis as the undeformed backbone axis.
# Reference physical displacement [x, y, z] is tracked as model tip [L + z, x, y].
L1, L2 = 0.2225, 0.2175
D1, D2 = 0.016, 0.014
STATIC_MODEL_SEGMENTS = 2
STATIC_MOTOR_COUNT = 6
ROBOT_MOTOR_COUNT = 6
# Cable numbering: 0..2 are long cables, 3..5 are short cables.
ACTIVE_MOTOR_INDICES = np.array([0, 1, 2, 3, 4, 5], dtype=int)

PCS_MAX_COMMAND_MM = 40.0
PCS_MAX_COMMAND_STEP_MM = 5.0
MOTOR_FACTOR = 1.225

# Weight order is model [x, y, z]. Physical trajectory z maps to model x.
PCS_STATIC_Q = np.diag([2e4, 2e3, 2e3])
PCS_STATIC_R = 1e-5
PCS_STATIC_R_SMOOTH = 1
PCS_STATIC_PROGRESS_REWARD = 1e-4
PCS_STATIC_SOLVER = "linearized"  # "linearized" or "nonlinear"
PCS_STATIC_MAX_NFEV = 40
PCS_STATIC_COMPUTE_FULL_RESIDUAL = False
PCS_STATIC_MAX_LENGTH_RESIDUAL_M = 1e-2
# PCS_STATIC_RESIDUAL_BLEND_FACTORS = np.array([1.0, 0.75, 0.5, 0.25, 0.1, 0.0])
PCS_STATIC_RESIDUAL_BLEND_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


PCS_BACKBONE_E = 0.875e6
PCS_BACKBONE_RHO = 1050.0
PCS_BACKBONE_NU = 0.47


@dataclass(frozen=True)
class PCSStaticMpcResult:
    command_m: np.ndarray
    q: np.ndarray
    lam: np.ndarray
    tip_xyz: np.ndarray
    cost: float
    success: bool
    message: str
    nfev: int
    solve_time_s: float
    residual_norm: float


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def create_log_path() -> Path:
    log_dir = PLOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"pcs_static_mpc_log_{timestamp}.txt"


def section_stiffness_from_diameter(diameter: float) -> np.ndarray:
    G = PCS_BACKBONE_E / (2.0 * (1.0 + PCS_BACKBONE_NU))
    params = PCSBeamParameters(
        E=PCS_BACKBONE_E,
        G=G,
        rho=PCS_BACKBONE_RHO,
        backbone_radius=0.5 * float(diameter),
    )
    return params.section_stiffness()


def make_static_model() -> PCSTwoSegmentsModel:
    if STATIC_MODEL_SEGMENTS % 2 != 0:
        raise ValueError("STATIC_MODEL_SEGMENTS must be even so each section has the same PCS resolution.")

    segments_per_section = STATIC_MODEL_SEGMENTS // 2
    section_stiffness = np.repeat(
        np.stack(
            [
                section_stiffness_from_diameter(D1),
                section_stiffness_from_diameter(D2),
            ],
            axis=0,
        ),
        segments_per_section,
        axis=0,
    )
    return PCSTwoSegmentsModel.with_uniform_section_cables(
        section_lengths=np.array([L1, L2], dtype=float),
        section_radii=0.5 * np.array([D1, D2], dtype=float),
        segments_per_section=segments_per_section,
        section_stiffness=section_stiffness,
        swap_section_cable_order=False,
        gravity=np.zeros(3, dtype=float),
    )


def validate_static_robot_dimensions(model: PCSSingleSegmentModel) -> None:
    if STATIC_MOTOR_COUNT != 6:
        raise ValueError(f"STATIC_MOTOR_COUNT must be 6 for a 2-section, 3-cable/section robot, got {STATIC_MOTOR_COUNT}.")
    if model.n_cables != STATIC_MOTOR_COUNT:
        raise ValueError(f"Static model has {model.n_cables} cables, but STATIC_MOTOR_COUNT is {STATIC_MOTOR_COUNT}.")
    if getattr(model, "swap_section_cable_order", False):
        raise ValueError("Static cable order must be 0..2 long cables and 3..5 short cables.")
    if ACTIVE_MOTOR_INDICES.shape != (STATIC_MOTOR_COUNT,):
        raise ValueError(
            f"ACTIVE_MOTOR_INDICES must have shape ({STATIC_MOTOR_COUNT},), got {ACTIVE_MOTOR_INDICES.shape}."
        )
    if np.any(ACTIVE_MOTOR_INDICES < 0) or np.any(ACTIVE_MOTOR_INDICES >= ROBOT_MOTOR_COUNT):
        raise ValueError(
            f"ACTIVE_MOTOR_INDICES must be in [0, {ROBOT_MOTOR_COUNT - 1}], got {ACTIVE_MOTOR_INDICES.tolist()}."
        )


def load_trajectory() -> tuple[np.ndarray, np.ndarray]:
    zd = sciIO.loadmat(str(TRAJ_DIR / "zd_proj.mat"))["zd_proj"]
    if zd.ndim != 2 or zd.shape[0] != 3:
        raise ValueError(f"Expected zd_proj to have shape (3, N), got {zd.shape}")

    t_list = sciIO.loadmat(str(TRAJ_DIR / "t_list.mat"))["t_list"].squeeze()
    return zd, t_list


def reference_window(desired_displacements: np.ndarray, step_index: int, horizon: int) -> np.ndarray:
    last_index = desired_displacements.shape[1] - 1
    indices = np.clip(np.arange(step_index, step_index + horizon), 0, last_index)
    return desired_displacements[:, indices].T


def model_axis_length(model: PCSSingleSegmentModel) -> float:
    return float(np.asarray(model.breakpoints, dtype=float)[-1])


def model_target_from_physical_displacement(
    desired_displacements: np.ndarray,
    model: PCSSingleSegmentModel,
) -> np.ndarray:
    desired = np.asarray(desired_displacements, dtype=float)
    if desired.ndim != 2 or desired.shape[1] != 3:
        raise ValueError(f"desired_displacements must have shape (horizon, 3), got {desired.shape}")

    target = np.empty_like(desired)
    target[:, 0] = model_axis_length(model) + desired[:, 2]
    target[:, 1] = desired[:, 0]
    target[:, 2] = desired[:, 1]
    return target


def physical_displacement_from_model_tip(tip_xyz: np.ndarray, model: PCSSingleSegmentModel) -> np.ndarray:
    tip = np.asarray(tip_xyz, dtype=float).reshape(3)
    return np.array([tip[1], tip[2], tip[0] - model_axis_length(model)], dtype=float)


def connect_sdk():
    sdk = load_lumo_sdk(BASE_DIR)
    sdk.Init()
    if sdk.Connect(IP_ADDRESS):
        print("LuMo SDK connected successfully")
    else:
        raise ConnectionError(f"Failed to connect LuMo SDK at {IP_ADDRESS}")
    return sdk


def static_options(verbose: bool = False) -> PCSLengthStaticOptions:
    return PCSLengthStaticOptions(
        static_solver=PCS_STATIC_SOLVER,
        max_nfev=PCS_STATIC_MAX_NFEV,
        verbose=verbose,
        compute_residual=PCS_STATIC_COMPUTE_FULL_RESIDUAL,
    )


def tip_xyz_jacobian_fd(model: PCSSingleSegmentModel, q: np.ndarray, eps: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    jac = np.zeros((3, model.dof), dtype=float)
    for i in range(model.dof):
        dq = np.zeros(model.dof, dtype=float)
        dq[i] = eps
        jac[:, i] = (model.tip_position(q + dq) - model.tip_position(q - dq)) / (2.0 * eps)
    return jac


def command_to_tip_jacobian(
    model: PCSSingleSegmentModel,
    q_start: np.ndarray,
    *,
    cable_reference_lengths: np.ndarray,
    command_prev_m: np.ndarray,
    fd_eps: float,
) -> np.ndarray:
    q_start = np.asarray(q_start, dtype=float)
    command_prev_m = np.asarray(command_prev_m, dtype=float)
    target_lengths = np.asarray(cable_reference_lengths, dtype=float) - command_prev_m
    A = model.cable_length_jacobian_fd(q_start, eps=fd_eps)
    Qg = model.gravity_generalized_force_fd(q_start, eps=fd_eps)
    r_top = model.K_pcs @ (q_start - model.q0) - Qg
    r_bot = model.cable_lengths(q_start) - target_lengths

    n = model.dof
    m = model.n_cables
    kkt = np.block([[model.K_pcs, A.T], [A, np.zeros((m, m), dtype=float)]])
    rhs = np.vstack(
        [
            -np.column_stack([r_top, np.zeros((n, m), dtype=float)]),
            -np.column_stack([r_bot, np.eye(m, dtype=float)]),
        ]
    )
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    dq_dcommand = sol[:n, 1:]
    return tip_xyz_jacobian_fd(model, q_start, eps=fd_eps) @ dq_dcommand


def solve_bounded_linear_mpc_step(
    *,
    tip_base: np.ndarray,
    tip_jac_u: np.ndarray,
    target_xyz: np.ndarray,
    prev_u: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float, bool, str]:
    command_size = prev_u.size
    q_weight = np.asarray(PCS_STATIC_Q, dtype=float)
    smooth_weight_m = float(PCS_STATIC_R_SMOOTH) * 1.0e6
    regularization = float(PCS_STATIC_R) + smooth_weight_m
    linear_offset = tip_base - tip_jac_u @ prev_u - target_xyz
    hessian = tip_jac_u.T @ q_weight @ tip_jac_u + regularization * np.eye(command_size)
    gradient = tip_jac_u.T @ q_weight @ linear_offset - smooth_weight_m * prev_u

    try:
        command = -np.linalg.solve(hessian, gradient)
        message = "bounded linear least-squares step"
    except np.linalg.LinAlgError:
        command = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        message = "bounded linear least-squares step via lstsq"

    command = np.clip(command, lower, upper)
    predicted_tip = tip_base + tip_jac_u @ (command - prev_u)
    error = predicted_tip - target_xyz
    du = command - prev_u
    cost = float(error.T @ q_weight @ error)
    cost += float(PCS_STATIC_R) * float(command @ command)
    cost += float(PCS_STATIC_R_SMOOTH) * float((1000.0 * du) @ (1000.0 * du))
    success = bool(np.all(np.isfinite(command)) and np.isfinite(cost))
    return command, cost, success, message


def solve_static_mpc(
    model: PCSSingleSegmentModel,
    ref_xyz: np.ndarray,
    *,
    q0: np.ndarray | None,
    lam0: np.ndarray | None,
    command_prev_m: np.ndarray | None,
    cable_reference_lengths: np.ndarray,
) -> PCSStaticMpcResult:
    t0 = time.perf_counter()
    refs = np.asarray(ref_xyz, dtype=float)
    if refs.ndim != 2 or refs.shape[1] != 3:
        raise ValueError(f"ref_xyz must have shape (horizon, 3), got {refs.shape}")

    horizon = refs.shape[0]
    command_size = model.n_cables
    q_start = model.initial_state() if q0 is None else np.asarray(q0, dtype=float).copy()
    lam_start = np.zeros(command_size, dtype=float) if lam0 is None else np.asarray(lam0, dtype=float).copy()
    prev_u = np.zeros(command_size, dtype=float) if command_prev_m is None else np.asarray(command_prev_m, dtype=float).copy()

    max_command_m = PCS_MAX_COMMAND_MM / 1000.0
    max_step_m = PCS_MAX_COMMAND_STEP_MM / 1000.0
    lower = np.maximum(-max_command_m, prev_u - max_step_m)
    upper = np.minimum(max_command_m, prev_u + max_step_m)
    opts = static_options(verbose=False)
    horizon_weights = np.linspace(1.0, 0.35, horizon, dtype=float)
    horizon_weights /= np.sum(horizon_weights)
    target_xyz = horizon_weights @ refs
    tip_base = model.tip_position(q_start)
    tip_jac_u = command_to_tip_jacobian(
        model,
        q_start,
        cable_reference_lengths=cable_reference_lengths,
        command_prev_m=prev_u,
        fd_eps=opts.fd_eps,
    )
    command, cost, optimizer_success, optimizer_message = solve_bounded_linear_mpc_step(
        tip_base=tip_base,
        tip_jac_u=tip_jac_u,
        target_xyz=target_xyz,
        prev_u=prev_u,
        lower=lower,
        upper=upper,
    )
    residual_limit = float(PCS_STATIC_MAX_LENGTH_RESIDUAL_M)
    q_weight = np.asarray(PCS_STATIC_Q, dtype=float)
    best_command = None
    best_point = None
    best_cost = float("inf")
    best_blend = 1.0
    best_residual = float("inf")
    fallback_command = None
    fallback_point = None
    fallback_cost = float("inf")
    fallback_blend = 0.0
    fallback_residual = float("inf")
    for blend in PCS_STATIC_RESIDUAL_BLEND_FACTORS:
        trial_command = prev_u + float(blend) * (command - prev_u)
        trial_point = solve_pcs_length_static_forward_point(
            model=model,
            cable_length_controls=trial_command,
            cable_reference_lengths=cable_reference_lengths,
            options=opts,
            q0=q_start,
            lam0=lam_start,
        )
        if not np.isfinite(trial_point.residual_norm):
            continue

        actual_error = model.tip_position(trial_point.q) - target_xyz
        trial_du = trial_command - prev_u
        trial_cost = float(actual_error.T @ q_weight @ actual_error)
        trial_cost += float(PCS_STATIC_R) * float(trial_command @ trial_command)
        trial_cost += float(PCS_STATIC_R_SMOOTH) * float((1000.0 * trial_du) @ (1000.0 * trial_du))
        trial_cost -= float(PCS_STATIC_PROGRESS_REWARD) * float(blend)
        residual_ok = trial_point.residual_norm <= residual_limit
        if residual_ok and trial_cost < best_cost:
            best_command = trial_command
            best_point = trial_point
            best_cost = trial_cost
            best_blend = float(blend)
            best_residual = float(trial_point.residual_norm)
        if fallback_point is None or trial_point.residual_norm < fallback_residual:
            fallback_command = trial_command
            fallback_point = trial_point
            fallback_cost = trial_cost
            fallback_blend = float(blend)
            fallback_residual = float(trial_point.residual_norm)

    if best_point is None:
        best_command = fallback_command if fallback_command is not None else prev_u.copy()
        best_point = fallback_point
        best_cost = fallback_cost
        best_blend = fallback_blend
        if best_point is None:
            best_point = solve_pcs_length_static_forward_point(
                model=model,
                cable_length_controls=best_command,
                cable_reference_lengths=cable_reference_lengths,
                options=opts,
                q0=q_start,
                lam0=lam_start,
            )
            best_error = model.tip_position(best_point.q) - target_xyz
            best_cost = float(best_error.T @ q_weight @ best_error)
        optimizer_success = False

    command = np.asarray(best_command, dtype=float)
    point = best_point
    cost = float(best_cost)
    if best_blend < 1.0:
        optimizer_message += f"; residual blend x{best_blend:.2f}"
    if point.residual_norm > residual_limit:
        optimizer_success = False
        optimizer_message += f"; residual limit exceeded {point.residual_norm:.3e}"

    return PCSStaticMpcResult(
        command_m=command.copy(),
        q=point.q.copy(),
        lam=point.cable_multipliers.copy(),
        tip_xyz=model.tip_position(point.q).copy(),
        cost=float(cost),
        success=bool(optimizer_success and point.success),
        message=f"outer: {optimizer_message}; inner: {point.message}",
        nfev=1,
        solve_time_s=float(time.perf_counter() - t0),
        residual_norm=float(point.residual_norm),
    )


def motor_command_from_static_command(command_m: np.ndarray) -> np.ndarray:
    motor_command = np.zeros(ROBOT_MOTOR_COUNT, dtype=float)
    motor_command[ACTIVE_MOTOR_INDICES] = np.asarray(command_m, dtype=float).reshape(STATIC_MOTOR_COUNT) * 1000.0
    return clamp_motor_positions(motor_command * MOTOR_FACTOR)


@fixed_cpu_time(target_cpu_seconds=CONTROL_PERIOD_SECONDS)
def run_control_step(
    sdk,
    socket,
    model: PCSSingleSegmentModel,
    step_index: int,
    desired_displacements: np.ndarray,
    q0: np.ndarray | None,
    lam0: np.ndarray | None,
    command_prev_m: np.ndarray | None,
    cable_reference_lengths: np.ndarray,
    reference,
    y_expm: np.ndarray,
    q_save: np.ndarray,
    lam_save: np.ndarray,
    u_save: np.ndarray,
    static_tip_save: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ref_displacement = reference_window(desired_displacements, step_index, MPC_HORIZON)
    ref_xyz = model_target_from_physical_displacement(ref_displacement, model)
    result = solve_static_mpc(
        model,
        ref_xyz,
        q0=q0,
        lam0=lam0,
        command_prev_m=command_prev_m,
        cable_reference_lengths=cable_reference_lengths,
    )

    motor_positions_mm = motor_command_from_static_command(result.command_m)
    print(
        f"PCS static MPC cost = {result.cost:.6e}, nfev = {result.nfev}, "
        f"success = {result.success}, solve time = {result.solve_time_s:.6f}s, "
        f"residual = {result.residual_norm:.6e}"
    )
    print(f"control action = {motor_positions_mm}")
    send_motor_command(socket, motor_positions_mm, solver_order=True)

    # Motion capture is temporarily disabled. Keep y_expm as zeros so result
    # files retain the same shape expected by plotting scripts.
    # disp_end, disp_mid, _ = capture_motion(sdk, reference)
    # y_expm[step_index, :] = np.hstack([disp_end, disp_mid])
    q_save[step_index, :] = result.q.reshape(-1)
    lam_save[step_index, :] = result.lam.reshape(-1)
    u_save[step_index, :] = np.asarray(motor_positions_mm, dtype=float)
    static_tip_save[step_index, :] = physical_displacement_from_model_tip(result.tip_xyz, model)
    return result.q, result.lam, result.command_m, result.solve_time_s


def save_results(
    y_expm: np.ndarray,
    t_list: np.ndarray,
    q_save: np.ndarray,
    lam_save: np.ndarray,
    u_save: np.ndarray,
    static_tip_save: np.ndarray,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / "y_expm_pcs_static", y_expm)
    np.save(RESULT_DIR / "t_expm_pcs_static", t_list)
    np.save(RESULT_DIR / "q_save_pcs_static", q_save)
    np.save(RESULT_DIR / "lambda_save_pcs_static", lam_save)
    np.save(RESULT_DIR / "u_save_pcs_static", u_save)
    np.save(RESULT_DIR / "static_tip_pcs_static", static_tip_save)


def run_experiment() -> None:
    sdk = connect_sdk()
    context = None
    socket = None

    try:
        model = make_static_model()
        validate_static_robot_dimensions(model)
        desired_displacements, t_list = load_trajectory()
        step_count = desired_displacements.shape[1]
        cable_reference_lengths = model.cable_lengths(model.initial_state())
        print(
            f"PCS static length model: segments = {model.N}, cables = {model.n_cables}, "
            f"solver = {PCS_STATIC_SOLVER}, horizon = {MPC_HORIZON}, "
            f"control period = {CONTROL_PERIOD_SECONDS:.3f}s"
        )

        y_expm = np.zeros((step_count, 6))
        q_save = np.zeros((step_count, model.dof))
        lam_save = np.zeros((step_count, model.n_cables))
        u_save = np.zeros((step_count, ROBOT_MOTOR_COUNT))
        static_tip_save = np.zeros((step_count, 3))

        # Motion capture is temporarily disabled.
        # reference = initialize_motion_reference(sdk)
        reference = None
        context, socket = create_publisher(MOTOR_ENDPOINT)
        print("ZeroMQ publisher started on /controller/setActuatorDate")

        q0 = None
        lam0 = None
        command_prev_m = None
        max_mpc_elapsed_time = 0.0
        for step_index in range(step_count):
            start_time = time.perf_counter()
            q0, lam0, command_prev_m, mpc_elapsed_time = run_control_step(
                sdk=sdk,
                socket=socket,
                model=model,
                step_index=step_index,
                desired_displacements=desired_displacements,
                q0=q0,
                lam0=lam0,
                command_prev_m=command_prev_m,
                cable_reference_lengths=cable_reference_lengths,
                reference=reference,
                y_expm=y_expm,
                q_save=q_save,
                lam_save=lam_save,
                u_save=u_save,
                static_tip_save=static_tip_save,
            )
            max_mpc_elapsed_time = max(max_mpc_elapsed_time, mpc_elapsed_time)
            elapsed_time = time.perf_counter() - start_time
            print(
                f"Step {step_index}: running time = {elapsed_time:.6f}s, "
                f"mpc time = {mpc_elapsed_time:.6f}s, "
                f"max mpc step time = {max_mpc_elapsed_time:.6f}s\n"
            )

        save_results(y_expm, t_list[:step_count], q_save, lam_save, u_save, static_tip_save)
        print(y_expm)

    finally:
        if socket is not None and context is not None:
            close_publisher(context, socket)
        if sdk is not None:
            sdk.Close()


def main() -> None:
    log_path = create_log_path()
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = Tee(sys.stdout, log_file)
        stderr_tee = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"PCS static MPC log file: {log_path}")
            run_experiment()


if __name__ == "__main__":
    main()
