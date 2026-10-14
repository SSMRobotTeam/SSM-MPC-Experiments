from dataclasses import dataclass, field
import time

import casadi as ca
import numpy as np


MOVE_WEIGHT = 1000.0


@dataclass
class DataSsmExperimentConfig:
    Np: int = 3
    Q: np.ndarray = field(default_factory=lambda: 1000.0 * np.eye(3))
    Qf: np.ndarray = field(default_factory=lambda: 1000.0 * np.eye(3))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.003] * 3 + [0.008] * 3))
    difference_step: float = 0.001
    move_weight: float = MOVE_WEIGHT


def move_penalty_quadratic(nominal, previous, weight, dt, factor):
    nominal = np.asarray(nominal, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if nominal.ndim != 2 or previous.shape != (nominal.shape[1],):
        raise ValueError('Control horizon and previous input dimensions do not agree.')
    if not np.isfinite(weight) or weight < 0 or dt <= 0 or (factor <= 0):
        raise ValueError('The move weight must be finite and nonnegative.')
    horizon, nu = nominal.shape
    count = horizon * nu
    D = np.eye(count)
    for j in range(1, horizon):
        D[j * nu:(j + 1) * nu, (j - 1) * nu:j * nu] = -np.eye(nu)
    anchor = np.zeros(count)
    anchor[:nu] = previous
    residual = D @ nominal.ravel() - anchor
    scale = dt * weight * factor ** 2
    return (2 * scale * (D.T @ D), 2 * scale * (D.T @ residual))


class DataSsmMpcController:

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or DataSsmExperimentConfig()
        if not np.isfinite(self.config.move_weight) or self.config.move_weight < 0:
            raise ValueError('The motor-increment weight must be finite and nonnegative.')
        self.previous_control = np.zeros(model.nu)
        self.eta = np.zeros(model.SSMDim)
        self.nominal = np.zeros((self.config.Np, model.nu))
        self.yk_ssm = model.output(self.eta)
        self.last_solve_time = 0.0
        self.lower = max(model.uMinR, model.sentMin / model.factor)
        self.upper = min(model.uMaxR, model.sentMax / model.factor)
        if self.lower >= self.upper or self.config.Np < 1:
            raise ValueError('Invalid MPC bounds or horizon.')
        count = model.nu * self.config.Np
        self.selectors = []
        for index in range(self.config.Np):
            selector = np.zeros((model.nu, count))
            selector[:, index * model.nu:(index + 1) * model.nu] = np.eye(model.nu)
            self.selectors.append(selector)
        self.solver = ca.conic('dssm_mpc', 'qpoases', {'h': ca.Sparsity.dense(count, count)}, {'printLevel': 'none', 'error_on_fail': True})

    def linearize(self, state, control):
        model = self.model
        epsilon = self.config.difference_step
        next_state = model.step(state, control)
        output = model.output(next_state)
        A = np.zeros((model.SSMDim, model.SSMDim))
        B = np.zeros((model.SSMDim, model.nu))
        output_state = np.zeros((3, model.SSMDim))
        output_input = np.zeros((3, model.nu))
        for index in range(model.SSMDim):
            perturbed = state.copy()
            perturbed[index] += epsilon
            advanced = model.step(perturbed, control)
            A[:, index] = (advanced - next_state) / epsilon
            output_state[:, index] = (model.output(advanced) - output) / epsilon
        for index in range(model.nu):
            perturbed = control.copy()
            perturbed[index] += epsilon
            advanced = model.step(state, perturbed)
            B[:, index] = (advanced - next_state) / epsilon
            output_input[:, index] = (model.output(advanced) - output) / epsilon
        return (A, B, output_state, output_input, next_state, output)

    def compute_next_command(self, references):
        cfg = self.config
        model = self.model
        references = np.asarray(references, dtype=float)
        if references.shape != (cfg.Np, 3) or not np.all(np.isfinite(references)):
            raise ValueError('MPC references must have shape (horizon, 3) and be finite.')
        count = cfg.Np * model.nu
        H = np.zeros((count, count))
        gradient = np.zeros(count)
        state = self.eta.copy()
        sensitivity = np.zeros((model.SSMDim, count))
        for index in range(cfg.Np):
            control = self.nominal[index]
            A, B, output_state, output_input, next_state, output = self.linearize(state, control)
            selector = self.selectors[index]
            output_sensitivity = output_state @ sensitivity + output_input @ selector
            weight = model.dt * cfg.Q
            if index == cfg.Np - 1:
                weight = weight + cfg.Qf
            error = output - references[index]
            H += 2 * output_sensitivity.T @ weight @ output_sensitivity
            gradient += 2 * output_sensitivity.T @ weight @ error
            H += 2 * model.dt * selector.T @ cfg.R @ selector
            gradient += 2 * model.dt * selector.T @ cfg.R @ control
            sensitivity = A @ sensitivity + B @ selector
            state = next_state
        move_H, move_gradient = move_penalty_quadratic(self.nominal, self.previous_control, cfg.move_weight, model.dt, model.factor)
        H += move_H
        gradient += move_gradient
        H = 0.5 * (H + H.T)
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(gradient)):
            raise RuntimeError('Non-finite MPC linearization.')
        lower = (self.lower - self.nominal).reshape(-1)
        upper = (self.upper - self.nominal).reshape(-1)
        started = time.perf_counter()
        solution = self.solver(h=ca.DM(H), g=ca.DM(gradient), lbx=ca.DM(lower), ubx=ca.DM(upper))
        self.last_solve_time = time.perf_counter() - started
        if not self.solver.stats().get('success', False):
            raise RuntimeError(f'MPC solver failed: {self.solver.stats().get('return_status')}')
        optimized = self.nominal + np.asarray(solution['x']).reshape(cfg.Np, model.nu)
        if not np.all(np.isfinite(optimized)):
            raise RuntimeError('Non-finite MPC solution.')
        if np.any(optimized < self.lower - 1e-06) or np.any(optimized > self.upper + 1e-06):
            raise RuntimeError('MPC solution violates the control bounds.')
        optimized = np.clip(optimized, self.lower, self.upper)
        control = optimized[0].copy()
        next_state = model.step(self.eta, control)
        output = model.output(next_state)
        if not np.all(np.isfinite(next_state)) or not np.all(np.isfinite(output)):
            raise RuntimeError('Non-finite model prediction.')
        self.previous_control = control.copy()
        self.eta = next_state
        self.yk_ssm = output
        self.nominal = np.vstack([optimized[1:], optimized[-1]])
        return model.factor * control


def plan_trajectory(model, desired, config=None):
    desired = np.asarray(desired, dtype=float)
    if desired.ndim != 2 or desired.shape[0] != 3 or desired.shape[1] < 2 or (not np.all(np.isfinite(desired))):
        raise ValueError('References must be a finite (3, K+1) matrix with K >= 1.')
    controller = DataSsmMpcController(model, config)
    steps = desired.shape[1] - 1
    commands = np.zeros((steps, model.nu))
    predictions = np.zeros((steps, 3))
    states = np.zeros((steps, model.SSMDim))
    compute_times = np.zeros(steps)
    solve_times = np.zeros(steps)
    for index in range(steps):
        references = np.array([desired[:, min(index + stage + 1, steps)] for stage in range(controller.config.Np)])
        started = time.perf_counter()
        commands[index] = controller.compute_next_command(references)
        compute_times[index] = time.perf_counter() - started
        solve_times[index] = controller.last_solve_time
        predictions[index] = controller.yk_ssm
        states[index] = controller.eta
    return {'ut_save': commands, 'y_ssm': predictions, 'pk_save': states, 'mpc_time': compute_times, 'solve_time': solve_times}
