from dataclasses import dataclass

import numpy as np

from controller.nmpc import NmpcConfig, suppress_native_output


@dataclass
class PcsNmpcResult:
    strains: np.ndarray
    tendon_delta: np.ndarray
    cost: float
    success: bool
    nfev: int
    model_error: float = 0.0


def nmpc_pcs_multi_shooting(
    robot,
    q0,
    ref_positions,
    cont: NmpcConfig,
    *,
    zero_strains=None,
    max_command_step_m: float | None = None,
    R_strain: float = 1e0,
    R_smooth: float = 1e0,
    R_bend: float = 1.0,
    R_twist: float = 1.0,
    R_axial: float = 1.0,
    R_shear: float = 1.0,
    max_nfev: int = 120,
) -> PcsNmpcResult:
    """Solve PCS NMPC with a CasADi NLP.

    Decision variables are the PCS strain trajectory Q and routed tendon
    command trajectory U. The PCS model constraint is enforced explicitly:
        U_k = tendon_lengths(Q_k) - tendon_lengths(Q_zero).
    The objective includes tip trajectory tracking, control energy, control
    smoothness, and structural strain regularization.
    """

    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("The PCS NMPC controller requires the Python package `casadi`.") from exc

    ref_positions = np.asarray(ref_positions, dtype=float)
    if ref_positions.ndim != 2 or ref_positions.shape[0] != 3:
        raise ValueError(f"ref_positions must have shape (3, Np), got {ref_positions.shape}")

    horizon = int(cont.Np)
    if ref_positions.shape[1] != horizon:
        raise ValueError(f"ref_positions must provide {horizon} columns, got {ref_positions.shape[1]}")

    q_prev = np.asarray(q0, dtype=float).reshape(len(robot.sections), robot.strain_size)
    q_zero = (
        np.zeros_like(q_prev)
        if zero_strains is None
        else np.asarray(zero_strains, dtype=float).reshape(q_prev.shape)
    )
    zero_lengths = robot.tendon_lengths(q_zero)
    prev_delta = robot.tendon_lengths(q_prev) - zero_lengths

    q_shape = q_prev.shape
    q_size = q_prev.size
    u_size = prev_delta.size
    q_tracking = np.asarray(_tracking_matrix(cont.Q), dtype=float)
    r_weight = max(float(cont.R), 0.0)
    u_lb = _control_bounds(cont.u_lb, u_size)
    u_ub = _control_bounds(cont.u_ub, u_size)
    strain_weights = np.tile(
        _strain_component_weights(robot.strain_size, R_bend, R_twist, R_axial, R_shear),
        len(robot.sections),
    )

    solver_type = cont.solver_type.lower()
    if solver_type == "qp":
        return _solve_pcs_linearized_qp_ca(
            ca=ca,
            robot=robot,
            q_prev=q_prev,
            q_zero=q_zero,
            zero_lengths=zero_lengths,
            prev_delta=prev_delta,
            ref_positions=ref_positions,
            q_tracking=q_tracking,
            r_weight=r_weight,
            u_lb=u_lb,
            u_ub=u_ub,
            strain_weights=strain_weights,
            horizon=horizon,
            max_command_step_m=max_command_step_m,
            R_strain=R_strain,
            R_smooth=R_smooth,
            qp_backend=cont.qp_backend,
        )
    if solver_type != "ipopt":
        raise ValueError(f"Unknown PCS MPC solver_type '{cont.solver_type}'. Use 'ipopt' or 'qp'.")

    Q = ca.SX.sym("Q", q_size, horizon)
    U = ca.SX.sym("U", u_size, horizon)

    obj = 0
    g = []
    lbg = []
    ubg = []
    last_u = ca.DM(prev_delta)

    for k in range(horizon):
        qk_flat = Q[:, k]
        uk = U[:, k]
        tip_position = _pcs_tip_position_ca(ca, robot, qk_flat)
        tendon_delta = _pcs_tendon_delta_ca(ca, robot, qk_flat, zero_lengths)
        position_error = tip_position - ca.DM(ref_positions[:, k])
        delta_u = uk - last_u

        obj += position_error.T @ ca.DM(q_tracking) @ position_error
        obj += 0.5 * r_weight * (uk.T @ uk)
        obj += float(R_smooth) * r_weight * (delta_u.T @ delta_u)
        weighted_strain = ca.DM(strain_weights) * qk_flat
        obj += float(R_strain) * r_weight * (weighted_strain.T @ weighted_strain)

        model_constraint = uk - tendon_delta
        g.append(model_constraint)
        lbg.extend([0.0] * u_size)
        ubg.extend([0.0] * u_size)

        if max_command_step_m is not None:
            g.append(delta_u)
            lbg.extend([-float(max_command_step_m)] * u_size)
            ubg.extend([float(max_command_step_m)] * u_size)

        last_u = uk

    opt_variables = ca.vertcat(
        ca.reshape(Q, q_size * horizon, 1),
        ca.reshape(U, u_size * horizon, 1),
    )
    g_vec = ca.vertcat(*g)

    lbx = np.full((q_size * horizon + u_size * horizon,), -np.inf)
    ubx = np.full_like(lbx, np.inf)
    lbx[q_size * horizon :] = np.tile(u_lb, horizon)
    ubx[q_size * horizon :] = np.tile(u_ub, horizon)

    q_guess = np.tile(q_prev.reshape(-1), (horizon, 1)).T
    u_guess = np.tile(np.clip(prev_delta, u_lb, u_ub), (horizon, 1)).T
    x0_guess = np.concatenate(
        [
            q_guess.reshape(-1, order="F"),
            u_guess.reshape(-1, order="F"),
        ]
    )

    solver = ca.nlpsol(
        "pcs_mpc_ipopt_solver",
        "ipopt",
        {"f": obj, "x": opt_variables, "g": g_vec},
        {
            "ipopt.print_level": 'none',
            "ipopt.max_iter": int(max_nfev),
            "ipopt.tol": 1e-7,
            "ipopt.acceptable_tol": 1e-6,
            "print_time": 0,
        },
    )
    sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=np.asarray(lbg), ubg=np.asarray(ubg))

    x_opt = np.asarray(sol["x"]).reshape(-1)
    q_opt = x_opt[: q_size * horizon].reshape((q_size, horizon), order="F")
    u_opt = x_opt[q_size * horizon :].reshape((u_size, horizon), order="F")
    first_q = q_opt[:, 0].reshape(q_shape)
    first_model_delta = robot.tendon_lengths(first_q) - zero_lengths
    model_error = float(np.linalg.norm(u_opt[:, 0] - first_model_delta, ord=np.inf))
    stats = solver.stats()

    return PcsNmpcResult(
        strains=first_q,
        tendon_delta=first_model_delta,
        cost=float(sol["f"]),
        success=bool(stats.get("success", False)),
        nfev=int(stats.get("iter_count", 0)),
        model_error=model_error,
    )


def _pcs_tip_position_ca(ca, robot, q_flat):
    q_units = _q_flat_to_units(q_flat, len(robot.sections), robot.strain_size)
    T = ca.SX.eye(4)
    for section, strain in zip(robot.sections, q_units):
        T = T @ _exp_se3_constant_strain_ca(ca, strain, float(section.length), robot.strain_size)
    return T[0:3, 3]


def _solve_pcs_linearized_qp_ca(
    *,
    ca,
    robot,
    q_prev,
    q_zero,
    zero_lengths,
    prev_delta,
    ref_positions,
    q_tracking,
    r_weight,
    u_lb,
    u_ub,
    strain_weights,
    horizon,
    max_command_step_m,
    R_strain,
    R_smooth,
    qp_backend,
) -> PcsNmpcResult:
    q_shape = q_prev.shape
    q_size = q_prev.size
    u_size = prev_delta.size
    q_linear = q_prev.reshape(-1)
    tip_base = robot.tip_position(q_prev)
    tendon_base = robot.tendon_lengths(q_prev) - zero_lengths
    J_tip = robot.position_jacobian(q_prev)
    J_tendon = robot.tendon_jacobian(q_prev)

    Q = ca.SX.sym("Q", q_size, horizon)
    U = ca.SX.sym("U", u_size, horizon)

    obj = 0
    g = []
    lbg = []
    ubg = []
    last_u = ca.DM(prev_delta)
    q_linear_dm = ca.DM(q_linear)

    for k in range(horizon):
        qk = Q[:, k]
        uk = U[:, k]
        dq = qk - q_linear_dm
        tip_linear = ca.DM(tip_base) + ca.DM(J_tip) @ dq
        tendon_linear = ca.DM(tendon_base) + ca.DM(J_tendon) @ dq
        position_error = tip_linear - ca.DM(ref_positions[:, k])
        delta_u = uk - last_u

        obj += position_error.T @ ca.DM(q_tracking) @ position_error
        obj += 0.5 * r_weight * (uk.T @ uk)
        obj += float(R_smooth) * r_weight * (delta_u.T @ delta_u)
        weighted_strain = ca.DM(strain_weights) * qk
        obj += float(R_strain) * r_weight * (weighted_strain.T @ weighted_strain)

        g.append(uk - tendon_linear)
        lbg.extend([0.0] * u_size)
        ubg.extend([0.0] * u_size)

        if max_command_step_m is not None:
            g.append(delta_u)
            lbg.extend([-float(max_command_step_m)] * u_size)
            ubg.extend([float(max_command_step_m)] * u_size)

        last_u = uk

    opt_variables = ca.vertcat(
        ca.reshape(Q, q_size * horizon, 1),
        ca.reshape(U, u_size * horizon, 1),
    )
    g_vec = ca.vertcat(*g)
    lbx = np.full((q_size * horizon + u_size * horizon,), -np.inf)
    ubx = np.full_like(lbx, np.inf)
    lbx[q_size * horizon :] = np.tile(u_lb, horizon)
    ubx[q_size * horizon :] = np.tile(u_ub, horizon)

    q_guess = np.tile(q_linear, (horizon, 1)).T
    u_guess = np.tile(np.clip(prev_delta, u_lb, u_ub), (horizon, 1)).T
    x0_guess = np.concatenate(
        [
            q_guess.reshape(-1, order="F"),
            u_guess.reshape(-1, order="F"),
        ]
    )

    options = {"print_time": False}
    is_qpoases = qp_backend.lower() == "qpoases"
    if is_qpoases:
        options["printLevel"] = "none"
    if is_qpoases:
        with suppress_native_output():
            solver = ca.qpsol(
                "pcs_mpc_qp_solver",
                qp_backend,
                {"x": opt_variables, "f": obj, "g": g_vec},
                options,
            )
            sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=np.asarray(lbg), ubg=np.asarray(ubg))
    else:
        solver = ca.qpsol(
            "pcs_mpc_qp_solver",
            qp_backend,
            {"x": opt_variables, "f": obj, "g": g_vec},
            options,
        )
        sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=np.asarray(lbg), ubg=np.asarray(ubg))

    x_opt = np.asarray(sol["x"]).reshape(-1)
    q_opt = x_opt[: q_size * horizon].reshape((q_size, horizon), order="F")
    u_opt = x_opt[q_size * horizon :].reshape((u_size, horizon), order="F")
    first_q = q_opt[:, 0].reshape(q_shape)
    first_model_delta = robot.tendon_lengths(first_q) - robot.tendon_lengths(q_zero)
    stats = solver.stats()
    return PcsNmpcResult(
        strains=first_q,
        tendon_delta=first_model_delta,
        cost=float(sol["f"]),
        success=bool(stats.get("success", True)),
        nfev=int(stats.get("iter_count", 0)),
        model_error=float(np.linalg.norm(u_opt[:, 0] - first_model_delta, ord=np.inf)),
    )


def _pcs_tendon_delta_ca(ca, robot, q_flat, zero_lengths):
    q_units = _q_flat_to_units(q_flat, len(robot.sections), robot.strain_size)
    tendon_lengths = _pcs_tendon_lengths_ca(ca, robot, q_units)
    return tendon_lengths - ca.DM(np.asarray(zero_lengths, dtype=float).reshape(-1))


def _q_flat_to_units(q_flat, section_count, strain_size):
    return [q_flat[i * strain_size : (i + 1) * strain_size] for i in range(section_count)]


def _exp_se3_constant_strain_ca(ca, strain, length, strain_size):
    w, v = _split_constant_strain_ca(ca, strain, strain_size)
    W = _hat_ca(ca, w)
    W2 = W @ W
    wn = ca.sqrt(ca.dot(w, w) + 1e-12)
    theta = wn * length

    R = ca.SX.eye(3) + ca.sin(theta) / wn * W + (1.0 - ca.cos(theta)) / (wn * wn) * W2
    V = (
        ca.SX.eye(3) * length
        + (1.0 - ca.cos(theta)) / (wn * wn) * W
        + (theta - ca.sin(theta)) / (wn * wn * wn) * W2
    )
    p = V @ v

    return ca.vertcat(
        ca.horzcat(R, p),
        ca.horzcat(0.0, 0.0, 0.0, 1.0),
    )


def _split_constant_strain_ca(ca, strain, strain_size):
    kx = strain[0]
    ky = strain[1]
    tau = strain[2]
    eps = strain[3]
    if int(strain_size) == 4:
        gamma_x = 0.0
        gamma_y = 0.0
    else:
        gamma_x = strain[4]
        gamma_y = strain[5]
    w = ca.vertcat(kx, ky, tau)
    v = ca.vertcat(gamma_x, gamma_y, 1.0 + eps)
    return w, v


def _hat_ca(ca, w):
    return ca.vertcat(
        ca.horzcat(0.0, -w[2], w[1]),
        ca.horzcat(w[2], 0.0, -w[0]),
        ca.horzcat(-w[1], w[0], 0.0),
    )


def _pcs_tendon_lengths_ca(ca, robot, q_units):
    long_count = len(robot.sections[0].tendons)
    long_tendons = [0.0 for _ in range(long_count)]

    for section_index, (section, strain) in enumerate(zip(robot.sections, q_units)):
        section_lengths = _section_tendon_lengths_ca(ca, section, section.tendons, strain, robot.strain_size)
        for tendon_index in range(long_count):
            long_tendons[tendon_index] = long_tendons[tendon_index] + section_lengths[tendon_index]

    short_count = len(robot.short_tendons)
    short_tendons = [0.0 for _ in range(short_count)]
    for section_index in range(robot.short_section_count):
        section_lengths = _section_tendon_lengths_ca(
            ca,
            robot.sections[section_index],
            robot.short_tendons,
            q_units[section_index],
            robot.strain_size,
        )
        for tendon_index in range(short_count):
            short_tendons[tendon_index] = short_tendons[tendon_index] + section_lengths[tendon_index]

    return ca.vertcat(*(long_tendons + short_tendons))


def _section_tendon_lengths_ca(ca, section, tendons, strain, strain_size):
    w, v = _split_constant_strain_ca(ca, strain, strain_size)
    kx = w[0]
    ky = w[1]
    tau = w[2]
    lengths = []

    for tendon in tendons:
        phi = float(tendon.angle)
        radius = float(tendon.mean_radius)
        shear_x = v[0] - tau * radius * np.sin(phi)
        shear_y = v[1] + tau * radius * np.cos(phi)
        axial_term = v[2] + radius * (kx * np.sin(phi) - ky * np.cos(phi))
        integrand = ca.sqrt(shear_x * shear_x + shear_y * shear_y + axial_term * axial_term + 1e-12)
        lengths.append(float(section.length) * integrand)

    return lengths


def _tracking_matrix(q_matrix):
    q = np.asarray(q_matrix, dtype=float)
    if q.ndim == 2:
        if q.shape[0] >= 3 and q.shape[1] >= 3:
            return q[:3, :3]
        out = np.eye(3)
        rows = min(3, q.shape[0])
        cols = min(3, q.shape[1])
        out[:rows, :cols] = q[:rows, :cols]
        return out
    q_vec = q.reshape(-1)
    diag = q_vec[:3] if q_vec.size >= 3 else np.pad(q_vec, (0, 3 - q_vec.size), constant_values=1.0)
    return np.diag(diag)


def _strain_component_weights(strain_size, R_bend, R_twist, R_axial, R_shear):
    if int(strain_size) == 4:
        return np.array([R_bend, R_bend, R_twist, R_axial], dtype=float)
    return np.array([R_bend, R_bend, R_twist, R_axial, R_shear, R_shear], dtype=float)


def _control_bounds(bounds, control_size):
    values = np.asarray(bounds, dtype=float).reshape(-1)
    if values.size < control_size:
        raise ValueError(f"Control bounds must contain at least {control_size} values.")
    return values[:control_size]
