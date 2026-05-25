from dataclasses import dataclass
from contextlib import contextmanager
import os
import sys

import numpy as np


@dataclass
class NmpcConfig:
    Q: np.ndarray
    R: float
    Np: int
    u_lb: np.ndarray
    u_ub: np.ndarray
    solver_type: str = "qp"
    qp_backend: str = "qpoases"
    use_fast_single_step: bool = True
    pos_end: np.ndarray | None = None
    pos_obstacle: np.ndarray | None = None
    avoid_node_positions: np.ndarray | None = None
    d_safe: float | None = None
    avoid_beta: float = 1e5 # 1e5
    avoid_weight: float = 0.6 # 0.6
    control_smooth_weight: float = 1e2


@contextmanager
def suppress_native_output():
    """Temporarily silence C/C++ library output to stdout and stderr."""
    stdout_fd = sys.__stdout__.fileno() if sys.__stdout__ is not None else 1
    stderr_fd = sys.__stderr__.fileno() if sys.__stderr__ is not None else 2
    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def nmpc_multi_shooting_zoh(
    x0,
    x_guess,
    s0_guess,
    u_guess,
    ref_uv,
    cont: NmpcConfig,
    x_num,
    controlnum,
    Ad,
    Bd,
    dd,
    Mhat,
    bhat,
    Hp,
    btilde,
    avoid_mhat=None,
    avoid_bhat=None,
):
    """CasADi Python port of NMPC_MultiShooting_CasADi_ZOH.m."""
    if cont.use_fast_single_step and int(cont.Np) == 1:
        return solve_single_step_qp(
            x0=x0,
            s0_guess=s0_guess,
            u_guess=u_guess,
            ref_uv=ref_uv,
            cont=cont,
            x_num=x_num,
            controlnum=controlnum,
            Ad=Ad,
            Bd=Bd,
            dd=dd,
            Mhat=Mhat,
            bhat=bhat,
            Hp=Hp,
            btilde=btilde,
        )

    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("The SSM NMPC controller requires the Python package `casadi`.") from exc

    Q = np.asarray(cont.Q, dtype=float)
    R = float(cont.R)
    Np = int(cont.Np)
    control_smooth_weight = float(cont.control_smooth_weight)

    U = ca.SX.sym("U", controlnum, Np)
    S0 = ca.SX.sym("S0", x_num, Np)
    X = ca.SX.sym("X", x_num, Np + 1)

    g = [X[:, 0] - ca.DM(x0)]
    obj = 0
    delta_u = [U[:, 0] - ca.DM(u_guess[:, 0])]
    delta_s0 = [S0[:, 0] - ca.DM(s0_guess[:, 0])]

    for i in range(Np - 1):
        delta_u.append(U[:, i + 1] - U[:, i])
        delta_s0.append(S0[:, i + 1] - S0[:, i])

    for i in range(Np):
        xk = X[:, i]
        xk1 = X[:, i + 1]
        s0k = S0[:, i]
        uk = U[:, i]

        xk1_calc = ca.DM(Ad) @ xk + ca.DM(Bd) @ s0k + ca.DM(dd)
        g.append(xk1 - xk1_calc)
        g.append(ca.DM(Hp) @ xk1 + ca.DM(btilde) + uk)

        yk = ca.DM(Mhat) @ xk1 + ca.DM(bhat)
        ref = ca.DM(ref_uv[:, i])
        obj += (ref - yk).T @ ca.DM(Q) @ (ref - yk)
        obj += 1e-1 * R * (s0k.T @ s0k)
        obj += 1e0 * R * (delta_s0[i].T @ delta_s0[i])
        obj += 0.5e0 * R * (uk.T @ uk)
        obj += control_smooth_weight * R * (delta_u[i].T @ delta_u[i])

    opt_variables = ca.vertcat(
        ca.reshape(X, x_num * (Np + 1), 1),
        ca.reshape(S0, x_num * Np, 1),
        ca.reshape(U, controlnum * Np, 1),
    )
    g_vec = ca.vertcat(*g)
    n_state = x_num * (2 * Np + 1)
    lbx = np.full((n_state + controlnum * Np,), -np.inf)
    ubx = np.full((n_state + controlnum * Np,), np.inf)
    lbx[n_state:] = np.asarray(cont.u_lb, dtype=float).reshape(-1)
    ubx[n_state:] = np.asarray(cont.u_ub, dtype=float).reshape(-1)

    x0_guess = np.concatenate(
        [
            np.asarray(x_guess).reshape(-1, order="F"),
            np.asarray(s0_guess).reshape(-1, order="F"),
            np.asarray(u_guess).reshape(-1, order="F"),
        ]
    )
    solver_type = cont.solver_type.lower()
    if solver_type == "ipopt":
        sol = solve_mpc_with_ipopt(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx)
    elif solver_type == "qp":
        sol = solve_mpc_with_qp(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx, cont.qp_backend)
    else:
        raise ValueError(f"Unknown MPC solver_type '{cont.solver_type}'. Use 'ipopt' or 'qp'.")

    x_opt = np.asarray(sol["x"]).reshape(-1)
    cost = float(sol["f"])
    xt = x_opt[: x_num * (Np + 1)].reshape((x_num, Np + 1), order="F")
    st_start = x_num * (Np + 1)
    st_end = st_start + x_num * Np
    st = x_opt[st_start:st_end].reshape((x_num, Np), order="F")
    ut = x_opt[st_end:].reshape((controlnum, Np), order="F")
    return xt[:, 1:], st, ut, cost


def nmpc_multi_shooting_zoh_avoid_soft(
    x0,
    x_guess,
    s0_guess,
    u_guess,
    ref_uv,
    cont: NmpcConfig,
    x_num,
    controlnum,
    Ad,
    Bd,
    dd,
    Mhat,
    bhat,
    Hp,
    btilde,
    avoid_mhat=None,
    avoid_bhat=None,
):
    """CasADi Python port of NMPC_MultiShooting_CasADi_ZOH_avoid_soft.m."""
    if cont.pos_end is None or cont.pos_obstacle is None or cont.d_safe is None:
        raise ValueError("Obstacle avoidance NMPC requires pos_end, pos_obstacle, and d_safe.")

    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("The obstacle-avoidance SSM NMPC controller requires `casadi`.") from exc

    Q = np.asarray(cont.Q, dtype=float)
    R = float(cont.R)
    Np = int(cont.Np)
    control_smooth_weight = float(cont.control_smooth_weight)
    pos_end = np.asarray(cont.pos_end, dtype=float).reshape(3)
    pos_obstacle = np.asarray(cont.pos_obstacle, dtype=float).reshape(3)

    U = ca.SX.sym("U", controlnum, Np)
    S0 = ca.SX.sym("S0", x_num, Np)
    X = ca.SX.sym("X", x_num, Np + 1)

    g = [X[:, 0] - ca.DM(x0)]
    obj = 0
    delta_u = [U[:, 0] - ca.DM(u_guess[:, 0])]
    delta_s0 = [S0[:, 0] - ca.DM(s0_guess[:, 0])]

    for i in range(Np - 1):
        delta_u.append(U[:, i + 1] - U[:, i])
        delta_s0.append(S0[:, i + 1] - S0[:, i])

    for i in range(Np):
        xk = X[:, i]
        xk1 = X[:, i + 1]
        s0k = S0[:, i]
        uk = U[:, i]

        xk1_calc = ca.DM(Ad) @ xk + ca.DM(Bd) @ s0k + ca.DM(dd)
        g.append(xk1 - xk1_calc)
        g.append(ca.DM(Hp) @ xk1 + ca.DM(btilde) + uk)

        yk = ca.DM(Mhat) @ xk1 + ca.DM(bhat)
        end_position = yk[0:3] + ca.DM(pos_end)
        obstacle_delta = end_position - ca.DM(pos_obstacle)
        dist_avoid = ca.dot(obstacle_delta, obstacle_delta)
        g_avoid = dist_avoid - float(cont.d_safe)

        ref = ca.DM(ref_uv[:, i])
        obj += (ref - yk).T @ ca.DM(Q) @ (ref - yk)
        obj += 1e-1 * R * (s0k.T @ s0k)
        obj += 1e1 * R * (uk.T @ uk)
        obj += float(cont.avoid_weight) * ca.exp(-float(cont.avoid_beta) * g_avoid)
        obj += 1e0 * R * (delta_s0[i].T @ delta_s0[i])
        obj += control_smooth_weight * R * (delta_u[i].T @ delta_u[i])

    opt_variables = ca.vertcat(
        ca.reshape(X, x_num * (Np + 1), 1),
        ca.reshape(S0, x_num * Np, 1),
        ca.reshape(U, controlnum * Np, 1),
    )
    g_vec = ca.vertcat(*g)
    n_state = x_num * (2 * Np + 1)
    lbx = np.full((n_state + controlnum * Np,), -np.inf)
    ubx = np.full((n_state + controlnum * Np,), np.inf)
    lbx[n_state:] = np.asarray(cont.u_lb, dtype=float).reshape(-1)
    ubx[n_state:] = np.asarray(cont.u_ub, dtype=float).reshape(-1)

    x0_guess = np.concatenate(
        [
            np.asarray(x_guess).reshape(-1, order="F"),
            np.asarray(s0_guess).reshape(-1, order="F"),
            np.asarray(u_guess).reshape(-1, order="F"),
        ]
    )
    solver_type = cont.solver_type.lower()
    if solver_type == "ipopt":
        sol = solve_mpc_with_ipopt(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx)
    elif solver_type == "qp":
        sol = solve_mpc_with_qp(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx, cont.qp_backend)
    else:
        raise ValueError(f"Unknown MPC solver_type '{cont.solver_type}'. Use 'ipopt' or 'qp'.")

    x_opt = np.asarray(sol["x"]).reshape(-1)
    cost = float(sol["f"])
    xt = x_opt[: x_num * (Np + 1)].reshape((x_num, Np + 1), order="F")
    st_start = x_num * (Np + 1)
    st_end = st_start + x_num * Np
    st = x_opt[st_start:st_end].reshape((x_num, Np), order="F")
    ut = x_opt[st_end:].reshape((controlnum, Np), order="F")
    return xt[:, 1:], st, ut, cost


def nmpc_multi_shooting_zoh_avoid_complex(
    x0,
    x_guess,
    s0_guess,
    u_guess,
    ref_uv,
    cont: NmpcConfig,
    x_num,
    controlnum,
    Ad,
    Bd,
    dd,
    Mhat,
    bhat,
    Hp,
    btilde,
    avoid_mhat=None,
    avoid_bhat=None,
):
    """SSM NMPC with hard avoidance constraints for multiple obstacle-nearest nodes."""
    if (
        cont.pos_obstacle is None
        or cont.avoid_node_positions is None
        or cont.d_safe is None
        or avoid_mhat is None
        or avoid_bhat is None
    ):
        raise ValueError(
            "Complex obstacle avoidance NMPC requires pos_obstacle, avoid_node_positions, "
            "d_safe, avoid_mhat, and avoid_bhat."
        )

    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("The complex obstacle-avoidance SSM NMPC controller requires `casadi`.") from exc

    Q = np.asarray(cont.Q, dtype=float)
    R = float(cont.R)
    Np = int(cont.Np)
    solver_type = cont.solver_type.lower()
    control_smooth_weight = float(cont.control_smooth_weight)
    pos_obstacles = np.asarray(cont.pos_obstacle, dtype=float).reshape(-1, 3)
    avoid_node_positions = np.asarray(cont.avoid_node_positions, dtype=float).reshape(-1, 3)
    avoid_mhat = np.asarray(avoid_mhat, dtype=float)
    avoid_bhat = np.asarray(avoid_bhat, dtype=float).reshape(-1)
    x_guess_for_avoid = (
        np.tile(np.asarray(x0, dtype=float).reshape(-1, 1), (1, Np + 1))
        if x_guess is None
        else np.asarray(x_guess, dtype=float).reshape(x_num, Np + 1)
    )

    obstacle_count = pos_obstacles.shape[0]
    if avoid_node_positions.shape[0] != obstacle_count:
        raise ValueError("pos_obstacle and avoid_node_positions must have the same number of rows.")
    if avoid_mhat.shape[0] != 3 * obstacle_count or avoid_bhat.size != 3 * obstacle_count:
        raise ValueError("avoid_mhat/avoid_bhat must provide 3 rows per obstacle-nearest node.")

    U = ca.SX.sym("U", controlnum, Np)
    S0 = ca.SX.sym("S0", x_num, Np)
    X = ca.SX.sym("X", x_num, Np + 1)

    g_equal = [X[:, 0] - ca.DM(x0)]
    g_avoid_constraints = []
    obj = 0
    delta_u = [U[:, 0] - ca.DM(u_guess[:, 0])]
    delta_s0 = [S0[:, 0] - ca.DM(s0_guess[:, 0])]

    for i in range(Np - 1):
        delta_u.append(U[:, i + 1] - U[:, i])
        delta_s0.append(S0[:, i + 1] - S0[:, i])

    for i in range(Np):
        xk = X[:, i]
        xk1 = X[:, i + 1]
        s0k = S0[:, i]
        uk = U[:, i]

        xk1_calc = ca.DM(Ad) @ xk + ca.DM(Bd) @ s0k + ca.DM(dd)
        g_equal.append(xk1 - xk1_calc)
        g_equal.append(ca.DM(Hp) @ xk1 + ca.DM(btilde) + uk)

        yk = ca.DM(Mhat) @ xk1 + ca.DM(bhat)
        ref = ca.DM(ref_uv[:, i])
        obj += (ref - yk).T @ ca.DM(Q) @ (ref - yk)
        obj += 1e-1 * R * (s0k.T @ s0k)
        obj += 1e1 * R * (uk.T @ uk)
        obj += 1e0 * R * (delta_s0[i].T @ delta_s0[i])
        obj += control_smooth_weight * R * (delta_u[i].T @ delta_u[i])

        for obstacle_idx in range(obstacle_count):
            row_start = 3 * obstacle_idx
            row_end = row_start + 3
            node_mhat = avoid_mhat[row_start:row_end, :]
            node_bhat = avoid_bhat[row_start:row_end]
            node_displacement = ca.DM(node_mhat) @ xk1 + ca.DM(node_bhat)
            node_position = node_displacement + ca.DM(avoid_node_positions[obstacle_idx])
            obstacle_delta = node_position - ca.DM(pos_obstacles[obstacle_idx])
            if solver_type == "qp":
                ref_displacement = node_mhat @ x_guess_for_avoid[:, i + 1] + node_bhat
                ref_delta = ref_displacement + avoid_node_positions[obstacle_idx] - pos_obstacles[obstacle_idx]
                ref_dist = np.linalg.norm(ref_delta)
                if ref_dist <= 1e-12:
                    ref_delta = avoid_node_positions[obstacle_idx] - pos_obstacles[obstacle_idx]
                    ref_dist = np.linalg.norm(ref_delta)
                if ref_dist <= 1e-12:
                    ref_delta = np.array([1.0, 0.0, 0.0])
                    ref_dist = 1.0
                # lbg >= 0 below enforces g_avoid >= 0, i.e. squared distance >= d_safe^2.
                g_avoid = (
                    float(ref_delta @ ref_delta)
                    + 2.0 * (ca.DM(ref_delta).T @ (obstacle_delta - ca.DM(ref_delta)))
                    - float(cont.d_safe) ** 2
                )
            else:
                dist_avoid = ca.dot(obstacle_delta, obstacle_delta)
                # lbg >= 0 below enforces g_avoid >= 0, i.e. squared distance >= d_safe^2.
                g_avoid = dist_avoid - float(cont.d_safe) ** 2
            g_avoid_constraints.append(g_avoid)

    opt_variables = ca.vertcat(
        ca.reshape(X, x_num * (Np + 1), 1),
        ca.reshape(S0, x_num * Np, 1),
        ca.reshape(U, controlnum * Np, 1),
    )
    g_vec = ca.vertcat(*(g_equal + g_avoid_constraints))
    n_equal_constraints = x_num + Np * (x_num + controlnum)
    n_avoid_constraints = Np * obstacle_count
    lbg = np.zeros(n_equal_constraints + n_avoid_constraints)
    ubg = np.zeros(n_equal_constraints + n_avoid_constraints)
    lbg[n_equal_constraints:] = 1e-9
    ubg[n_equal_constraints:] = np.inf

    n_state = x_num * (2 * Np + 1)
    lbx = np.full((n_state + controlnum * Np,), -np.inf)
    ubx = np.full((n_state + controlnum * Np,), np.inf)
    lbx[n_state:] = np.asarray(cont.u_lb, dtype=float).reshape(-1)
    ubx[n_state:] = np.asarray(cont.u_ub, dtype=float).reshape(-1)

    x0_guess = np.concatenate(
        [
            np.asarray(x_guess).reshape(-1, order="F"),
            np.asarray(s0_guess).reshape(-1, order="F"),
            np.asarray(u_guess).reshape(-1, order="F"),
        ]
    )
    if solver_type == "ipopt":
        sol = solve_mpc_with_ipopt(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx, lbg, ubg)
    elif solver_type == "qp":
        sol = solve_mpc_with_qp(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx, cont.qp_backend, lbg, ubg)
    else:
        raise ValueError(f"Unknown MPC solver_type '{cont.solver_type}'. Use 'ipopt' or 'qp'.")

    x_opt = np.asarray(sol["x"]).reshape(-1)
    cost = float(sol["f"])
    xt = x_opt[: x_num * (Np + 1)].reshape((x_num, Np + 1), order="F")
    st_start = x_num * (Np + 1)
    st_end = st_start + x_num * Np
    st = x_opt[st_start:st_end].reshape((x_num, Np), order="F")
    ut = x_opt[st_end:].reshape((controlnum, Np), order="F")
    return xt[:, 1:], st, ut, cost


def solve_mpc_with_ipopt(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx, lbg=0, ubg=0):
    solver = ca.nlpsol(
        "mpc_ipopt_solver",
        "ipopt",
        {"f": obj, "x": opt_variables, "g": g_vec},
        {"ipopt.print_level": 0, "print_time": 0},
    )
    sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    if not stats.get("success", False):
        status = stats.get("return_status", "unknown")
        raise RuntimeError(f"IPOPT failed to solve NMPC problem: {status}")
    return sol


def solve_mpc_with_qp(ca, obj, opt_variables, g_vec, x0_guess, lbx, ubx, qp_backend, lbg=0, ubg=0):
    if qp_backend.lower() == "qpoases":
        options = {"printLevel": "none", "print_time": False}
        with suppress_native_output():
            solver = ca.qpsol(
                "mpc_qp_solver",
                qp_backend,
                {"x": opt_variables, "f": obj, "g": g_vec},
                options,
            )
            sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            stats = solver.stats()
            if not stats.get("success", False):
                status = stats.get("return_status", "unknown")
                raise RuntimeError(f"{qp_backend} failed to solve NMPC QP: {status}")
            return sol

    solver = ca.qpsol(
        "mpc_qp_solver",
        qp_backend,
        {"x": opt_variables, "f": obj, "g": g_vec},
        {},
    )
    sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    if not stats.get("success", False):
        status = stats.get("return_status", "unknown")
        raise RuntimeError(f"{qp_backend} failed to solve NMPC QP: {status}")
    return sol


def solve_single_step_qp(
    x0,
    s0_guess,
    u_guess,
    ref_uv,
    cont: NmpcConfig,
    x_num,
    controlnum,
    Ad,
    Bd,
    dd,
    Mhat,
    bhat,
    Hp,
    btilde,
):
    """Fast closed-form QP for Np=1.

    This removes the per-step CasADi graph construction cost. For Np=1:
      X1 = Ad*x0 + Bd*S0 + dd
      U  = -Hp*X1 - btilde
    so the optimization can be reduced to a small quadratic problem in S0.
    """
    x0 = np.asarray(x0, dtype=float).reshape(x_num)
    s0_ref = np.asarray(s0_guess, dtype=float).reshape(x_num, -1)[:, 0]
    u_ref = np.asarray(u_guess, dtype=float).reshape(controlnum, -1)[:, 0]
    ref = np.asarray(ref_uv, dtype=float).reshape(-1, 1)[:, 0]

    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float)
    dd = np.asarray(dd, dtype=float).reshape(x_num)
    Mhat = np.asarray(Mhat, dtype=float)
    bhat = np.asarray(bhat, dtype=float).reshape(-1)
    Hp = np.asarray(Hp, dtype=float)
    btilde = np.asarray(btilde, dtype=float).reshape(controlnum)

    x_base = Ad @ x0 + dd
    y_base = Mhat @ x_base + bhat
    y_s = Mhat @ Bd
    u_base = -(Hp @ x_base + btilde)
    u_s = -(Hp @ Bd)

    Q = np.asarray(cont.Q, dtype=float)
    R = float(cont.R)
    control_smooth_weight = float(cont.control_smooth_weight)
    identity = np.eye(x_num)
    control_hessian_weight = 1.0 + 2.0 * control_smooth_weight
    control_gradient_weight = 2.0 * control_smooth_weight
    hessian = (
        2.0 * (y_s.T @ Q @ y_s)
        + 2.2 * R * identity
        + control_hessian_weight * R * (u_s.T @ u_s)
    )
    gradient = (
        2.0 * y_s.T @ Q @ (y_base - ref)
        - 2.0 * R * s0_ref
        + R * u_s.T @ (control_hessian_weight * u_base - control_gradient_weight * u_ref)
    )

    try:
        s0 = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        s0 = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

    u = u_base + u_s @ s0
    lower = np.asarray(cont.u_lb, dtype=float).reshape(-1)[:controlnum]
    upper = np.asarray(cont.u_ub, dtype=float).reshape(-1)[:controlnum]
    if np.any(u < lower) or np.any(u > upper):
        s0, u = solve_single_step_qp_with_bounds(s0, hessian, gradient, u_base, u_s, lower, upper)

    x1 = x_base + Bd @ s0
    cost = 0.5 * float(s0.T @ hessian @ s0) + float(gradient.T @ s0)
    return x1.reshape(x_num, 1), s0.reshape(x_num, 1), u.reshape(controlnum, 1), cost


def solve_single_step_qp_with_bounds(s0_init, hessian, gradient, u_base, u_s, lower, upper):
    try:
        from scipy.optimize import minimize
    except ImportError:
        u = np.clip(u_base + u_s @ s0_init, lower, upper)
        return s0_init, u

    constraints = [
        {"type": "ineq", "fun": lambda s, row=row, ub=ub: ub - (u_base[row] + u_s[row, :] @ s)}
        for row, ub in enumerate(upper)
    ]
    constraints.extend(
        {"type": "ineq", "fun": lambda s, row=row, lb=lb: (u_base[row] + u_s[row, :] @ s) - lb}
        for row, lb in enumerate(lower)
    )
    result = minimize(
        lambda s: 0.5 * s.T @ hessian @ s + gradient.T @ s,
        s0_init,
        jac=lambda s: hessian @ s + gradient,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 20, "ftol": 1e-8, "disp": False},
    )
    s0 = result.x if result.success else s0_init
    u = np.clip(u_base + u_s @ s0, lower, upper)
    return s0, u
