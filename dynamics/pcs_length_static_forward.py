from __future__ import annotations

"""Length-controlled static forward solves for the PCS comparison model.

The prescribed input is the cable shortening vector
    u = l_ref - l_target.
The PCS equilibrium is solved with cable-length constraints:
    K(q-q0) - Q_g(q) + A(q)^T lambda = 0
    l(q) - (l_ref - u) = 0
where lambda is the cable-length constraint multiplier.
"""

from dataclasses import dataclass
import time

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from .pcs_tension_static_optimizer import PCSSingleSegmentModel, PCSTwoSegmentsModel

Array = npt.NDArray[np.float64]
PCSLengthModel = PCSSingleSegmentModel | PCSTwoSegmentsModel


@dataclass(frozen=True)
class PCSLengthStaticOptions:
    """Options for PCS length-controlled forward replay."""

    static_solver: str = "linearized"  # "linearized" or "nonlinear"
    fd_eps: float = 1.0e-6
    residual_scale_length: float = 1.0e3
    max_nfev: int = 120
    xtol: float = 1.0e-9
    ftol: float = 1.0e-9
    gtol: float = 1.0e-9
    verbose: bool = True
    compute_residual: bool = True


@dataclass(frozen=True)
class PCSLengthStaticPointResult:
    cable_length_controls: Array
    cable_reference_lengths: Array
    cable_target_lengths: Array
    q: Array
    cable_lengths: Array
    cable_multipliers: Array
    tip_yz: Array
    cable_length_error: Array
    success: bool
    message: str
    residual_norm: float
    solve_time_s: float
    nfev: int


@dataclass(frozen=True)
class PCSLengthStaticTrajectoryResult:
    cable_length_controls: Array
    cable_reference_lengths: Array
    cable_target_lengths: Array
    q: Array
    cable_lengths: Array
    cable_multipliers: Array
    tip_yz: Array
    cable_length_error: Array
    success: Array
    messages: tuple[str, ...]
    residual_norm: Array
    solve_time_s: Array
    point_results: tuple[PCSLengthStaticPointResult, ...]


def _solve_linearized_kkt(
    *,
    model: PCSLengthModel,
    q_start: Array,
    cable_target_lengths: Array,
    fd_eps: float,
    compute_residual: bool = True,
) -> tuple[Array, Array, float]:
    """One local KKT solve around q_start."""
    q_start = np.asarray(q_start, dtype=float)
    l_start = model.cable_lengths(q_start)
    A = model.cable_length_jacobian_fd(q_start, eps=fd_eps)
    Qg = model.gravity_generalized_force_fd(q_start, eps=fd_eps)

    r_top = model.K_pcs @ (q_start - model.q0) - Qg
    r_bot = l_start - cable_target_lengths

    n = model.dof
    m = model.n_cables
    kkt = np.block(
        [
            [model.K_pcs, A.T],
            [A, np.zeros((m, m), dtype=float)],
        ]
    )
    rhs = -np.concatenate([r_top, r_bot])
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    dq = sol[:n]
    lam = sol[n:]
    q_new = q_start + dq

    if not compute_residual:
        length_error = model.cable_lengths(q_new) - cable_target_lengths
        return q_new, lam, float(np.linalg.norm(length_error))

    residual = pcs_length_static_residual(
        model=model,
        q=q_new,
        lam=lam,
        cable_target_lengths=cable_target_lengths,
        fd_eps=fd_eps,
    )
    return q_new, lam, float(np.linalg.norm(residual))


def pcs_length_static_residual(
    *,
    model: PCSLengthModel,
    q: Array,
    lam: Array,
    cable_target_lengths: Array,
    fd_eps: float,
) -> Array:
    """Unscaled PCS KKT residual for length-controlled static equilibrium."""
    q = np.asarray(q, dtype=float)
    lam = np.asarray(lam, dtype=float)
    target = np.asarray(cable_target_lengths, dtype=float)
    A = model.cable_length_jacobian_fd(q, eps=fd_eps)
    Qg = model.gravity_generalized_force_fd(q, eps=fd_eps)
    r_top = model.K_pcs @ (q - model.q0) - Qg + A.T @ lam
    r_bot = model.cable_lengths(q) - target
    return np.concatenate([r_top, r_bot])


def solve_pcs_length_static_forward_point(
    *,
    model: PCSLengthModel,
    cable_length_controls: Array,
    cable_reference_lengths: Array | None = None,
    options: PCSLengthStaticOptions | None = None,
    q0: Array | None = None,
    lam0: Array | None = None,
) -> PCSLengthStaticPointResult:
    """Solve one PCS static equilibrium with cable length controls."""
    t0 = time.perf_counter()
    opts = PCSLengthStaticOptions() if options is None else options

    u = np.asarray(cable_length_controls, dtype=float).copy()
    if u.shape != (model.n_cables,):
        raise ValueError(f"cable_length_controls must have shape ({model.n_cables},), got {u.shape}")
    l_ref = model.cable_lengths(model.initial_state()) if cable_reference_lengths is None else np.asarray(cable_reference_lengths, dtype=float).copy()
    if l_ref.shape != (model.n_cables,):
        raise ValueError(f"cable_reference_lengths must have shape ({model.n_cables},), got {l_ref.shape}")
    target_lengths = l_ref - u

    q_start = model.initial_state() if q0 is None else np.asarray(q0, dtype=float).copy()
    if q_start.shape != (model.dof,):
        raise ValueError(f"q0 must have shape ({model.dof},), got {q_start.shape}")
    lam_start = np.zeros(model.n_cables, dtype=float) if lam0 is None else np.asarray(lam0, dtype=float).copy()
    if lam_start.shape != (model.n_cables,):
        raise ValueError(f"lam0 must have shape ({model.n_cables},), got {lam_start.shape}")

    solver = str(opts.static_solver).lower()
    if solver == "linearized":
        q_sol, lam_sol, residual_norm = _solve_linearized_kkt(
            model=model,
            q_start=q_start,
            cable_target_lengths=target_lengths,
            fd_eps=opts.fd_eps,
            compute_residual=opts.compute_residual,
        )
        success = True
        message = "linearized PCS length-controlled static KKT"
        nfev = 1
    elif solver == "nonlinear":
        scale = float(opts.residual_scale_length)

        def packed_residual(x: Array) -> Array:
            q = x[: model.dof]
            lam = x[model.dof :]
            A = model.cable_length_jacobian_fd(q, eps=opts.fd_eps)
            Qg = model.gravity_generalized_force_fd(q, eps=opts.fd_eps)
            r_top = model.K_pcs @ (q - model.q0) - Qg + A.T @ lam
            r_bot = scale * (model.cable_lengths(q) - target_lengths)
            return np.concatenate([r_top, r_bot])

        x0 = np.concatenate([q_start, lam_start])
        res = least_squares(
            packed_residual,
            x0,
            max_nfev=int(opts.max_nfev),
            xtol=float(opts.xtol),
            ftol=float(opts.ftol),
            gtol=float(opts.gtol),
        )
        q_sol = res.x[: model.dof]
        lam_sol = res.x[model.dof :]
        residual_norm = float(np.linalg.norm(pcs_length_static_residual(
            model=model,
            q=q_sol,
            lam=lam_sol,
            cable_target_lengths=target_lengths,
            fd_eps=opts.fd_eps,
        )))
        success = bool(res.success)
        message = str(res.message)
        nfev = int(res.nfev)
    else:
        raise ValueError("static_solver must be 'linearized' or 'nonlinear'")

    lengths = model.cable_lengths(q_sol)
    return PCSLengthStaticPointResult(
        cable_length_controls=u.copy(),
        cable_reference_lengths=l_ref.copy(),
        cable_target_lengths=target_lengths.copy(),
        q=q_sol.copy(),
        cable_lengths=lengths.copy(),
        cable_multipliers=lam_sol.copy(),
        tip_yz=model.tip_position(q_sol)[1:3].copy(),
        cable_length_error=lengths - target_lengths,
        success=success,
        message=message,
        residual_norm=residual_norm,
        solve_time_s=float(time.perf_counter() - t0),
        nfev=nfev,
    )


def solve_pcs_length_static_forward_sequence(
    *,
    model: PCSLengthModel,
    cable_length_controls_sequence: Array,
    cable_reference_lengths: Array | None = None,
    options: PCSLengthStaticOptions | None = None,
    q0: Array | None = None,
) -> PCSLengthStaticTrajectoryResult:
    """Replay a cable shortening sequence through PCS static equilibrium."""
    U = np.asarray(cable_length_controls_sequence, dtype=float)
    if U.ndim != 2 or U.shape[1] != model.n_cables:
        raise ValueError(f"cable_length_controls_sequence must have shape (n_points,{model.n_cables}), got {U.shape}")
    opts = PCSLengthStaticOptions() if options is None else options
    l_ref = model.cable_lengths(model.initial_state()) if cable_reference_lengths is None else np.asarray(cable_reference_lengths, dtype=float).copy()
    if l_ref.shape != (model.n_cables,):
        raise ValueError(f"cable_reference_lengths must have shape ({model.n_cables},), got {l_ref.shape}")
    if U.shape[0] == 0:
        return PCSLengthStaticTrajectoryResult(
            cable_length_controls=U.copy(),
            cable_reference_lengths=l_ref.copy(),
            cable_target_lengths=np.empty((model.n_cables, 0), dtype=float),
            q=np.empty((model.dof, 0), dtype=float),
            cable_lengths=np.empty((model.n_cables, 0), dtype=float),
            cable_multipliers=np.empty((model.n_cables, 0), dtype=float),
            tip_yz=np.empty((0, 2), dtype=float),
            cable_length_error=np.empty((model.n_cables, 0), dtype=float),
            success=np.empty((0,), dtype=bool),
            messages=(),
            residual_norm=np.empty((0,), dtype=float),
            solve_time_s=np.empty((0,), dtype=float),
            point_results=(),
        )

    point_results: list[PCSLengthStaticPointResult] = []
    q_start = model.initial_state() if q0 is None else np.asarray(q0, dtype=float).copy()
    lam_start = np.zeros(model.n_cables, dtype=float)

    for i, u in enumerate(U):
        result = solve_pcs_length_static_forward_point(
            model=model,
            cable_length_controls=u,
            cable_reference_lengths=l_ref,
            options=opts,
            q0=q_start,
            lam0=lam_start,
        )
        point_results.append(result)
        q_start = result.q
        lam_start = result.cable_multipliers
        if opts.verbose:
            print(
                f"PCS length point {i + 1}/{U.shape[0]}: "
                f"err_len_mm={1.0e3 * np.linalg.norm(result.cable_length_error):.4f}, "
                f"tip_yz_mm={np.array2string(1.0e3 * result.tip_yz, precision=3)}, "
                f"time_s={result.solve_time_s:.3f}"
            )

    return PCSLengthStaticTrajectoryResult(
        cable_length_controls=U.copy(),
        cable_reference_lengths=l_ref.copy(),
        cable_target_lengths=np.column_stack([r.cable_target_lengths for r in point_results]),
        q=np.column_stack([r.q for r in point_results]),
        cable_lengths=np.column_stack([r.cable_lengths for r in point_results]),
        cable_multipliers=np.column_stack([r.cable_multipliers for r in point_results]),
        tip_yz=np.vstack([r.tip_yz for r in point_results]),
        cable_length_error=np.column_stack([r.cable_length_error for r in point_results]),
        success=np.asarray([r.success for r in point_results], dtype=bool),
        messages=tuple(r.message for r in point_results),
        residual_norm=np.asarray([r.residual_norm for r in point_results], dtype=float),
        solve_time_s=np.asarray([r.solve_time_s for r in point_results], dtype=float),
        point_results=tuple(point_results),
    )
