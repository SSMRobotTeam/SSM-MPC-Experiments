from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import scipy.io as sciIO
from scipy.linalg import expm
from scipy.optimize import least_squares

from controller.cable_constraints import cable_length_double_part, constraint_hz_part
from controller.nmpc import NmpcConfig, nmpc_multi_shooting_zoh
from dynamics.SSM.matlab_data import SSM_DIR
from dynamics.SSM.ssm_dynamics import (
    auto_red_dyn,
    jacobian_auto_red_dyn,
    jacobian_w_map,
    reduced_to_full,
)


@dataclass
class SsmExperimentConfig:
    dt: float
    tf: float
    td: np.ndarray
    zd: np.ndarray
    Np: int
    Q: np.ndarray
    R: float
    u_lb: np.ndarray
    u_ub: np.ndarray
    Node_List: object
    dist0: np.ndarray
    opt_W: float = 1.25
    solver_type: str = "ipopt"
    qp_backend: str = "qpoases"
    use_fast_single_step: bool = True
    nmpc_solver: object = nmpc_multi_shooting_zoh
    pos_end: np.ndarray | None = None
    pos_obstacle: np.ndarray | None = None
    avoid_node_positions: np.ndarray | None = None
    d_safe: float | None = None
    avoid_beta: float = 1e5
    avoid_weight: float = 0.5
    control_smooth_weight: float = 1e3
    motor_safety_limit: float = 40.0


@dataclass
class SsmFeedbackState:
    t: float
    k: int
    pk: np.ndarray
    S0_opt: np.ndarray
    ut_opt: np.ndarray


def initialize_actuator_state():
    return {
        "posi": np.zeros(6),
        "posi_hist": np.zeros((1, 6)),
        "veloc": np.zeros(6),
        "yk_ssm": np.zeros(9),
    }


def feedback_residual(yk, pk, w0_for_fb, p_guess, opt_w):
    y_cal = reduced_to_full(pk, w0_for_fb, None, 1).reshape(-1)
    return np.concatenate([y_cal - np.asarray(yk).reshape(-1), np.sqrt(opt_w) * (pk - p_guess)])


class SsmMpcController:
    """Python port of SSM_MPC_Actuator.m."""

    def __init__(
        self,
        config: SsmExperimentConfig,
        au_data,
        w0_for_out,
        w0_for_cables,
        w0_for_feedback,
        feedback_state: SsmFeedbackState,
        w0_for_avoid_nodes=None,
        factor: float = 1.0,
        feedback_enabled: bool = False,
        save_feedback_each_step: bool = False,
        state_dir: Path = SSM_DIR,
    ):
        self.config = config
        self.au_data = au_data
        self.w0_for_out = w0_for_out
        self.w0_for_cables = w0_for_cables
        self.w0_for_feedback = w0_for_feedback
        self.w0_for_avoid_nodes = w0_for_avoid_nodes
        self.feedback_state = feedback_state
        self.factor = factor
        self.feedback_enabled = feedback_enabled
        self.save_feedback_each_step = save_feedback_each_step
        self.state_dir = Path(state_dir)
        self.posi = np.zeros(6)
        self.posi_save = []
        self.yk_ssm = np.zeros(6)
        self.last_feedback_residual_norm = None
        self.last_feedback_accepted = False
        self.last_feedback_time = 0.0
        self.x_num = np.asarray(feedback_state.pk, dtype=float).reshape(-1).size
        self.identity = np.eye(self.x_num)
        self.nmpc_config = NmpcConfig(
            Q=config.Q,
            R=config.R,
            Np=config.Np,
            u_lb=config.u_lb,
            u_ub=config.u_ub,
            solver_type=config.solver_type,
            qp_backend=config.qp_backend,
            use_fast_single_step=config.use_fast_single_step,
            pos_end=config.pos_end,
            pos_obstacle=config.pos_obstacle,
            avoid_node_positions=config.avoid_node_positions,
            d_safe=config.d_safe,
            avoid_beta=config.avoid_beta,
            avoid_weight=config.avoid_weight,
            control_smooth_weight=config.control_smooth_weight,
        )

    @property
    def feedback_path(self) -> Path:
        return self.state_dir / "ssm_feedback_py.mat"

    def _apply_measurement_feedback(self, pk, measurement_mm, feedback=False):
        """Apply the measurement feedback block from SSM_MPC_Actuator.m lines 55-75."""
        self.last_feedback_residual_norm = None
        self.last_feedback_accepted = False
        self.last_feedback_time = 0.0

        if not feedback:
            return pk
        if measurement_mm is None:
            raise ValueError("measurement_mm is required when feedback=True")

        start_time = time.perf_counter()
        zk_meas = np.asarray(measurement_mm, dtype=float).reshape(9) / 1000.0
        result = least_squares(
            lambda p: feedback_residual(zk_meas, p, self.w0_for_feedback, pk, self.config.opt_W),
            pk,
            method="lm",
            max_nfev=3,
        )
        self.last_feedback_time = time.perf_counter() - start_time
        residual_norm = np.linalg.norm(result.fun[:9], 2)
        self.last_feedback_residual_norm = residual_norm
        if residual_norm < 1e-2:
            self.last_feedback_accepted = True
            return result.x
        return pk

    @staticmethod
    def _left_solve(a_matrix, rhs):
        try:
            return np.linalg.solve(a_matrix, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(a_matrix, rhs, rcond=None)[0]

    def _discretize_linear_model(self, a_matrix, d_vector):
        ad = expm(self.config.dt * a_matrix)
        ad_minus_eye = ad - self.identity
        rhs = np.column_stack([ad_minus_eye, ad_minus_eye @ d_vector])
        solved = self._left_solve(a_matrix, rhs)
        return ad, solved[:, : self.x_num], solved[:, self.x_num]

    @staticmethod
    def _linearize_w_map(w_map, pk):
        w0 = reduced_to_full(pk, w_map, None, 1)[:, 0]
        wp = jacobian_w_map(w_map, pk, w0.size, w0)
        b = -wp @ pk + w0
        return w0, wp, b

    def _linearize_avoid_nodes(self, pk):
        if self.w0_for_avoid_nodes is None:
            return None, None
        _, avoid_mhat, avoid_bhat = self._linearize_w_map(self.w0_for_avoid_nodes, pk)
        obstacle_count = 0 if self.config.pos_obstacle is None else np.asarray(self.config.pos_obstacle).reshape(-1, 3).shape[0]
        expected_rows = 3 * obstacle_count
        if expected_rows and (avoid_mhat.shape[0] != expected_rows or avoid_bhat.size != expected_rows):
            raise ValueError(
                "avoid_mhat/avoid_bhat shape mismatch: expected "
                f"{expected_rows} rows for {obstacle_count} obstacle-nearest nodes, "
                f"got {avoid_mhat.shape[0]} and {avoid_bhat.size}."
            )
        return avoid_mhat, avoid_bhat

    def compute_next_command(self, measurement_mm=None, feedback=None):
        cont = self.config
        state = self.feedback_state
        feedback_flag = self.feedback_enabled if feedback is None else bool(feedback)

        pk = self._apply_measurement_feedback(
            np.asarray(state.pk, dtype=float).reshape(-1),
            measurement_mm,
            feedback=feedback_flag,
        )
        zk_outdof, wp_out, bhat_full = self._linearize_w_map(self.w0_for_out, pk)
        zk_cables, wp_cables, _ = self._linearize_w_map(self.w0_for_cables, pk)

        if self.nmpc_config.use_fast_single_step and int(cont.Np) == 1:
            p_guess = None
            s0_guess = state.S0_opt
            u_guess = state.ut_opt
        else:
            p_guess = np.tile(pk.reshape(-1, 1), (1, cont.Np + 1))
            s0_guess = np.column_stack([state.S0_opt[:, 1:], state.S0_opt[:, -1]])
            u_guess = np.column_stack([state.ut_opt[:, 1:], state.ut_opt[:, -1]])

        ref_uv = self._reference_window(state.t, state.k)

        r0 = auto_red_dyn(pk, self.au_data).reshape(-1)
        rp = jacobian_auto_red_dyn(self.au_data, pk, r0)
        a_matrix = rp
        d_vector = -rp @ pk + r0

        ad, bd, dd = self._discretize_linear_model(a_matrix, d_vector)

        mhat = np.vstack([wp_out[0:3, :], wp_out[6:9, :]])
        bhat = np.concatenate([bhat_full[0:3], bhat_full[6:9]])
        avoid_mhat, avoid_bhat = self._linearize_avoid_nodes(pk)

        if self.config.avoid_node_positions is not None and (avoid_mhat is None or avoid_bhat is None):
            raise ValueError("Obstacle avoidance NMPC requires w0_for_avoid_nodes to build avoid_mhat/avoid_bhat.")

        d0 = cable_length_double_part(zk_cables.reshape(-1), cont.Node_List)
        hz = constraint_hz_part(zk_cables.reshape(-1), cont.Node_List)
        hp = hz @ wp_cables
        btilde = -hp @ pk + d0 - cont.dist0

        pt_opt, s0_opt, ut_opt, _ = cont.nmpc_solver(
            pk,
            p_guess,
            s0_guess,
            u_guess,
            ref_uv,
            self.nmpc_config,
            pk.size,
            6,
            ad,
            bd,
            dd,
            mhat,
            bhat,
            hp,
            btilde,
            avoid_mhat,
            avoid_bhat,
        )

        next_pk = pt_opt[:, 0]
        next_u = ut_opt[:, 0]
        commanded_posi = next_u * 1000.0 * self.factor
        self.posi = np.clip(commanded_posi, -cont.motor_safety_limit, cont.motor_safety_limit)

        self.yk_ssm = zk_outdof[:6] * 1000.0
        self.feedback_state = SsmFeedbackState(
            t=state.t + cont.dt,
            k=state.k + 1,
            pk=next_pk,
            S0_opt=s0_opt,
            ut_opt=ut_opt,
        )
        self.posi_save.append(self.posi.copy())
        if self.save_feedback_each_step:
            self.save_feedback_state()
        return self.posi

    def _reference_window(self, t, k):
        cont = self.config
        if int(cont.Np) == 1 and k < cont.zd.shape[1] - 1:
            yref = cont.zd[:, k]
            dyref = (cont.zd[:, k + 1] - yref) / cont.dt
            return np.concatenate([yref, dyref]).reshape(-1, 1)

        refs = []
        for i in range(cont.Np):
            t_i = t + i * cont.dt
            if t_i + (cont.Np - 1) * cont.dt >= cont.tf or k + i >= cont.zd.shape[1] - 1:
                yref = np.array([np.interp(t_i, cont.td, row) for row in cont.zd])
                yref_1 = np.array([np.interp(t_i + cont.dt, cont.td, row) for row in cont.zd])
                dyref = (yref_1 - yref) / cont.dt
            else:
                yref = cont.zd[:, k + i]
                dyref = (cont.zd[:, k + i + 1] - cont.zd[:, k + i]) / cont.dt
            refs.append(np.concatenate([yref, dyref]))
        return np.column_stack(refs)

    def save_feedback_state(self):
        sciIO.savemat(
            self.feedback_path,
            {
                "ssm_feedback": {
                    "t": self.feedback_state.t,
                    "k": self.feedback_state.k,
                    "pk": self.feedback_state.pk,
                    "S0_opt": self.feedback_state.S0_opt,
                    "ut_opt": self.feedback_state.ut_opt,
                }
            },
        )
