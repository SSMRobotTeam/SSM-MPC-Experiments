#!/usr/bin/python
import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import scipy.io as sciIO
from scipy.linalg import block_diag, expm

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from controller.nmpc import NmpcConfig, nmpc_multi_shooting_zoh
from dynamics.SSM.matlab_data import (
    clone_struct_array,
    get_field,
    loadmat,
    subset_mapping_rows,
    to_dense,
)
from dynamics.SSM.ssm_dynamics import (
    auto_red_dyn,
    jacobian_auto_red_dyn,
    jacobian_w_map,
    reduced_to_full,
)
from utils.sdk_utils import load_lumo_sdk
from utils.timing_utils import fixed_cpu_time
from zmq_message.zmq_control import close_publisher, create_publisher, send_motor_command


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results"
LOG_DIR = RESULT_DIR / "logs"
MODEL_DIRS = {
    "pod6": SCRIPT_DIR / "POD6model",
    "pod26": SCRIPT_DIR / "POD26model",
    "ssm6": SCRIPT_DIR / "SSM6model",
}
MATLAB_TRAJ_DIR = SCRIPT_DIR / "Diamond_SSM_Matlab"
SUT_TRAJ_DIR = BASE_DIR / "Diamond_PLOT" / "Diamond_SUT_PLOT" / "Diamond_DATA_comp_SUT_Trajs"
TRAJECTORY_CHOICES = ["symbol_s", "symbol_u", "symbol_t", "spiral3d"]
TRAJECTORY_ALIASES = {
    "s": "symbol_s",
    "u": "symbol_u",
    "t": "symbol_t",
    "traj_symbol_s": "symbol_s",
    "traj_symbol_u": "symbol_u",
    "traj_symbol_t": "symbol_t",
    "symbol_s": "symbol_s",
    "symbol_u": "symbol_u",
    "symbol_t": "symbol_t",
    "spiral3d": "spiral3d",
}
MOTOR_ENDPOINT = "tcp://localhost:5555"
IP_ADDRESS = "127.0.0.1"

MODEL = "ssm6"  # choose one: "pod6", "pod26", or "ssm6"
TRAJECTORY_NAME = "spiral3d"  # choose one: "symbol_s", "symbol_u", "symbol_t", or "spiral3d"
MEASUREMENT_ENABLED = False
BASE_RIGID_ID = 0
MID_RIGID_ID = 1
CHECKPOINT_RIGID_ID = 2
END_MARKER_ID = 100037
CONTROL_PERIOD_SECONDS = 0.075
HORIZON = 3

RUN_STEPS = 120
MOTOR_FACTOR = 1.0
MOTOR_SAFETY_LIMIT = 40.0
MPC_SOLVER_TYPE = "qp"
QP_BACKEND = "qpoases"
USE_FAST_SINGLE_STEP_QP = False

SSM_Q_WEIGHT = 5e3 * np.eye(6)
SSM_R_WEIGHT = 1e-1

POD_Q_WEIGHT = block_diag(np.diag([1e5, 1e5, 1e5]), np.zeros((3, 3)))
POD_R_WEIGHT = 1e-1

CONTROL_SMOOTH_WEIGHT = 1e1


class BufferedTeeLogger:
    def __init__(self, stream):
        self.stream = stream
        self.messages = []

    def write(self, message):
        self.stream.write(message)
        self.messages.append(message)

    def flush(self):
        self.stream.flush()

    def save(self, log_dir: Path) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        beijing_now = datetime.now(timezone(timedelta(hours=8)))
        log_path = log_dir / f"Diamond_MPC_log_{beijing_now:%Y%m%d_%H%M}.txt"
        if log_path.exists():
            index = 2
            while True:
                candidate = log_dir / f"Diamond_MPC_log_{beijing_now:%Y%m%d_%H%M}_{index:02d}.txt"
                if not candidate.exists():
                    log_path = candidate
                    break
                index += 1
        log_path.write_text("".join(self.messages), encoding="utf-8")
        return log_path


@dataclass
class DiamondExperimentConfig:
    dt: float
    tf: float
    td: np.ndarray
    zd: np.ndarray
    Np: int
    Q: np.ndarray
    R: float
    u_lb: np.ndarray
    u_ub: np.ndarray
    node_list: np.ndarray
    dist0: np.ndarray
    solver_type: str = MPC_SOLVER_TYPE
    qp_backend: str = QP_BACKEND
    use_fast_single_step: bool = USE_FAST_SINGLE_STEP_QP
    control_smooth_weight: float = CONTROL_SMOOTH_WEIGHT
    motor_safety_limit: float = MOTOR_SAFETY_LIMIT


@dataclass
class DiamondFeedbackState:
    t: float
    k: int
    x: np.ndarray
    s0_opt: np.ndarray
    u_opt: np.ndarray


def _as_node_list(value) -> np.ndarray:
    return np.asarray(value, dtype=object).reshape(-1)


def diamond_cable_length(x, node_list) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    nodes = _as_node_list(node_list)
    fixed_node = np.asarray(nodes[-1], dtype=float).reshape(-1)[:3]
    dist = np.zeros(4)
    for cable_idx in range(4):
        node = np.asarray(nodes[cable_idx], dtype=float).reshape(-1)
        dofs = node[3:6].astype(int) - 1
        moving_xyz = node[:3] + x[dofs]
        dist[cable_idx] = np.linalg.norm(fixed_node - moving_xyz, 2)
    return dist


def diamond_hz_constraint(x, node_list, include_velocity=True) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    nodes = _as_node_list(node_list)
    fixed_node = np.asarray(nodes[-1], dtype=float).reshape(-1)[:3]
    hx = np.zeros((4, x.size))
    for cable_idx in range(4):
        node = np.asarray(nodes[cable_idx], dtype=float).reshape(-1)
        dofs = node[3:6].astype(int) - 1
        moving_xyz = node[:3] + x[dofs]
        delta_xyz = moving_xyz - fixed_node
        norm_delta = np.linalg.norm(delta_xyz, 2)
        if norm_delta <= 1e-12:
            raise ValueError(f"Diamond cable {cable_idx + 1} has zero distance to fixed node.")
        hx[cable_idx, dofs] = delta_xyz / norm_delta
    if include_velocity:
        return np.hstack([hx, np.zeros_like(hx)])
    return hx


def diamond_cable_length_compact(cable_xyz_displacements, node_list) -> np.ndarray:
    x = np.asarray(cable_xyz_displacements, dtype=float).reshape(4, 3)
    nodes = _as_node_list(node_list)
    fixed_node = np.asarray(nodes[-1], dtype=float).reshape(-1)[:3]
    dist = np.zeros(4)
    for cable_idx in range(4):
        node = np.asarray(nodes[cable_idx], dtype=float).reshape(-1)
        moving_xyz = node[:3] + x[cable_idx]
        dist[cable_idx] = np.linalg.norm(fixed_node - moving_xyz, 2)
    return dist


def diamond_hz_constraint_compact(cable_xyz_displacements, node_list) -> np.ndarray:
    x = np.asarray(cable_xyz_displacements, dtype=float).reshape(4, 3)
    nodes = _as_node_list(node_list)
    fixed_node = np.asarray(nodes[-1], dtype=float).reshape(-1)[:3]
    hx = np.zeros((4, 12))
    for cable_idx in range(4):
        node = np.asarray(nodes[cable_idx], dtype=float).reshape(-1)
        moving_xyz = node[:3] + x[cable_idx]
        delta_xyz = moving_xyz - fixed_node
        norm_delta = np.linalg.norm(delta_xyz, 2)
        if norm_delta <= 1e-12:
            raise ValueError(f"Diamond cable {cable_idx + 1} has zero distance to fixed node.")
        hx[cable_idx, 3 * cable_idx : 3 * cable_idx + 3] = delta_xyz / norm_delta
    return hx


def cable_dofs_from_node_list(node_list) -> np.ndarray:
    dofs = []
    for cable_idx in range(4):
        node = np.asarray(_as_node_list(node_list)[cable_idx], dtype=float).reshape(-1)
        dofs.extend(node[3:6].astype(int).tolist())
    return np.asarray(dofs, dtype=int)


def normalize_trajectory_name(name: str) -> str:
    key = name.lower()
    if key not in TRAJECTORY_ALIASES:
        choices = ", ".join(TRAJECTORY_CHOICES)
        raise ValueError(f"Unknown trajectory '{name}'. Choose one of: {choices}")
    return TRAJECTORY_ALIASES[key]


def load_symbol_trajectory(symbol: str, steps: int, dt: float, traj_dir: Path = MATLAB_TRAJ_DIR) -> tuple[np.ndarray, np.ndarray]:
    suffix = symbol.upper()
    x_path = traj_dir / f"x_{suffix}.mat"
    y_path = traj_dir / f"y_{suffix}.mat"
    if not x_path.exists() or not y_path.exists():
        return load_symbol_trajectory_from_zzd(symbol, steps, dt)

    x_data = np.asarray(loadmat(x_path)[f"x_{suffix}"], dtype=float).reshape(-1)
    y_data = np.asarray(loadmat(y_path)[f"y_{suffix}"], dtype=float).reshape(-1)
    available_steps = min(x_data.size, y_data.size)
    step_count = min(int(steps), available_steps)
    if step_count <= 0:
        raise ValueError(f"Trajectory symbol {suffix} has no usable samples.")

    # MATLAB uses k = round(t/dt). With Python's 0-based arrays this corresponds
    # to the first step_count samples for td = 0:dt:...
    td = np.arange(step_count, dtype=float) * dt
    zd = np.zeros((3, step_count))
    zd[0, :] = x_data[:step_count]
    zd[1, :] = y_data[:step_count]
    return td, zd / 1000.0


def load_symbol_trajectory_from_zzd(symbol: str, steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    suffix = symbol.upper()
    zzd_path = SUT_TRAJ_DIR / f"0806-comp-{suffix}-symbol" / "data_SSM" / "zzd.mat"
    if not zzd_path.exists():
        raise FileNotFoundError(
            "Diamond symbol trajectory files are missing. Expected either "
            f"{MATLAB_TRAJ_DIR / f'x_{suffix}.mat'} and {MATLAB_TRAJ_DIR / f'y_{suffix}.mat'}, "
            f"or fallback {zzd_path}."
        )

    zzd = np.asarray(loadmat(zzd_path)["zzd"], dtype=float)
    if zzd.ndim != 2 or zzd.shape[0] != 3:
        raise ValueError(f"Expected {zzd_path} to contain a 3xN variable named zzd.")
    step_count = min(int(steps), zzd.shape[1])
    if step_count <= 0:
        raise ValueError(f"Trajectory symbol {suffix} has no usable samples.")
    td = np.arange(step_count, dtype=float) * dt
    return td, zzd[:, :step_count] / 1000.0


def make_spiral3d_trajectory(td: np.ndarray, radius_max_mm: float, height_mm: float, omega: float) -> np.ndarray:
    total_time = max(float(td[-1]), 1e-9)
    radius = radius_max_mm * (td / total_time)
    zd = np.zeros((3, td.size))
    zd[0, :] = 1e-3 * radius * np.cos(omega * td)
    zd[1, :] = 1e-3 * radius * np.sin(omega * td)
    zd[2, :] = 1e-3 * ((height_mm / total_time) * td + 7.5)
    return zd


def make_trajectory(
    name: str,
    steps: int,
    dt: float,
    spiral_radius_mm: float = 12.0,
    spiral_height_mm: float = 0.0,
    spiral_omega: float = 2.0 * np.pi / 4.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    key = normalize_trajectory_name(name)
    if key.startswith("symbol_"):
        td, zd = load_symbol_trajectory(key[-1], steps, dt)
        return td, zd, key

    td = np.arange(int(steps), dtype=float) * dt
    return td, make_spiral3d_trajectory(td, spiral_radius_mm, spiral_height_mm, spiral_omega), key


def reference_window(zd, td, t, k, dt, horizon):
    if int(horizon) == 1 and k < zd.shape[1] - 1:
        yref = zd[:, k]
        dyref = (zd[:, k + 1] - yref) / dt
        return np.concatenate([yref, dyref]).reshape(-1, 1)

    refs = []
    tf = float(td[-1])
    for i in range(horizon):
        t_i = t + i * dt
        if t_i + (horizon - 1) * dt >= tf or k + i >= zd.shape[1] - 1:
            yref = np.array([np.interp(t_i, td, row) for row in zd])
            yref_1 = np.array([np.interp(t_i + dt, td, row) for row in zd])
            dyref = (yref_1 - yref) / dt
        else:
            yref = zd[:, k + i]
            dyref = (zd[:, k + i + 1] - zd[:, k + i]) / dt
        refs.append(np.concatenate([yref, dyref]))
    return np.column_stack(refs)


def left_solve(a_matrix, rhs):
    try:
        return np.linalg.solve(a_matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a_matrix, rhs, rcond=None)[0]


def discretize_linear_model(a_matrix, d_vector, dt):
    x_num = a_matrix.shape[0]
    ad = expm(dt * a_matrix)
    ad_minus_eye = ad - np.eye(x_num)
    rhs = np.column_stack([ad_minus_eye, ad_minus_eye @ d_vector])
    solved = left_solve(a_matrix, rhs)
    return ad, solved[:, :x_num], solved[:, x_num]


class DiamondPodMpcController:
    def __init__(self, config, ar, phi_r, outdof_endeff, feedback_state, factor=MOTOR_FACTOR):
        self.config = config
        self.ar = np.asarray(ar, dtype=float)
        self.phi_r = np.asarray(phi_r, dtype=float)
        self.outdof_endeff = np.asarray(outdof_endeff, dtype=int).reshape(-1)
        self.feedback_state = feedback_state
        self.factor = factor
        self.full_map = block_diag(self.phi_r, self.phi_r)
        self.x_num = self.ar.shape[0]
        self.n_full = self.phi_r.shape[0]
        self.nmpc_config = self._build_nmpc_config()
        self.yk_model = np.zeros(6)

    def _build_nmpc_config(self):
        return NmpcConfig(
            Q=self.config.Q,
            R=self.config.R,
            Np=self.config.Np,
            u_lb=self.config.u_lb,
            u_ub=self.config.u_ub,
            solver_type=self.config.solver_type,
            qp_backend=self.config.qp_backend,
            use_fast_single_step=self.config.use_fast_single_step,
            control_smooth_weight=self.config.control_smooth_weight,
        )

    def compute_next_command(self):
        state = self.feedback_state
        x0 = np.asarray(state.x, dtype=float).reshape(-1)
        p_guess = np.tile(x0.reshape(-1, 1), (1, self.config.Np + 1))
        s0_guess = state.s0_opt
        u_guess = state.u_opt
        ref_uv = reference_window(self.config.zd, self.config.td, state.t, state.k, self.config.dt, self.config.Np)

        ad, bd, dd = discretize_linear_model(self.ar, np.zeros(self.x_num), self.config.dt)
        rows = np.concatenate([self.outdof_endeff, self.outdof_endeff + self.n_full]) - 1
        mhat = self.full_map[rows, :]
        bhat = np.zeros(6)

        full_z = self.full_map @ x0
        hz = diamond_hz_constraint(full_z[: self.n_full], self.config.node_list, include_velocity=True)
        hp = hz @ self.full_map
        d0 = diamond_cable_length(full_z[: self.n_full], self.config.node_list)
        btilde = -hp @ x0 + d0 - self.config.dist0

        xt_opt, s0_opt, u_opt, _ = nmpc_multi_shooting_zoh(
            x0,
            p_guess,
            s0_guess,
            u_guess,
            ref_uv,
            self.nmpc_config,
            self.x_num,
            4,
            ad,
            bd,
            dd,
            mhat,
            bhat,
            hp,
            btilde,
        )
        self.feedback_state = DiamondFeedbackState(
            t=state.t + self.config.dt,
            k=state.k + 1,
            x=xt_opt[:, 0],
            s0_opt=s0_opt,
            u_opt=u_opt,
        )
        self.yk_model = mhat @ xt_opt[:, 0] + bhat
        return np.clip(u_opt[:, 0] * 1000.0 * self.factor, -self.config.motor_safety_limit, self.config.motor_safety_limit)


class DiamondSsmMpcController:
    def __init__(self, config, au_data, w0, w0_for_out, w0_for_cables, feedback_state, n_full, factor=MOTOR_FACTOR):
        self.config = config
        self.au_data = au_data
        self.w0 = w0
        self.w0_for_out = w0_for_out
        self.w0_for_cables = w0_for_cables
        self.feedback_state = feedback_state
        self.n_full = int(n_full)
        self.factor = factor
        self.x_num = np.asarray(get_field(au_data, "lamd")).reshape(-1).size
        self.identity = np.eye(self.x_num)
        self.yk_model = np.zeros(6)
        self.nmpc_config = NmpcConfig(
            Q=config.Q,
            R=config.R,
            Np=config.Np,
            u_lb=config.u_lb,
            u_ub=config.u_ub,
            solver_type=config.solver_type,
            qp_backend=config.qp_backend,
            use_fast_single_step=config.use_fast_single_step,
            control_smooth_weight=config.control_smooth_weight,
        )

    def compute_next_command(self):
        state = self.feedback_state
        pk = np.asarray(state.x, dtype=float).reshape(-1)
        ref_uv = reference_window(self.config.zd, self.config.td, state.t, state.k, self.config.dt, self.config.Np)
        r0 = auto_red_dyn(pk, self.au_data).reshape(-1)
        rp = jacobian_auto_red_dyn(self.au_data, pk, r0)
        ad, bd, dd = discretize_linear_model(rp, -rp @ pk + r0, self.config.dt)

        zk_out, wp_out, bhat_out = self._linearize_w_map(self.w0_for_out, pk)
        zk_cables, wp_cables, _ = self._linearize_w_map(self.w0_for_cables, pk)

        hz = diamond_hz_constraint_compact(zk_cables.reshape(-1), self.config.node_list)
        hp = hz @ wp_cables
        d0 = diamond_cable_length_compact(zk_cables.reshape(-1), self.config.node_list)
        btilde = -hp @ pk + d0 - self.config.dist0

        p_guess = np.tile(pk.reshape(-1, 1), (1, self.config.Np + 1))
        xt_opt, s0_opt, u_opt, _ = nmpc_multi_shooting_zoh(
            pk,
            p_guess,
            state.s0_opt,
            state.u_opt,
            ref_uv,
            self.nmpc_config,
            self.x_num,
            4,
            ad,
            bd,
            dd,
            wp_out,
            bhat_out,
            hp,
            btilde,
        )
        self.feedback_state = DiamondFeedbackState(
            t=state.t + self.config.dt,
            k=state.k + 1,
            x=xt_opt[:, 0],
            s0_opt=s0_opt,
            u_opt=u_opt,
        )
        self.yk_model = reduced_to_full(xt_opt[:, 0], self.w0_for_out, None, 1).reshape(-1)
        return np.clip(u_opt[:, 0] * 1000.0 * self.factor, -self.config.motor_safety_limit, self.config.motor_safety_limit)

    @staticmethod
    def _linearize_w_map(w_map, pk):
        w0 = reduced_to_full(pk, w_map, None, 1)[:, 0]
        wp = jacobian_w_map(w_map, pk, w0.size, w0)
        b = -wp @ pk + w0
        return w0, wp, b


def build_config(model_data, trajectory_name, steps, dt, horizon, spiral_radius_mm, spiral_height_mm, spiral_omega, control_limit):
    td, zd, _ = make_trajectory(
        trajectory_name,
        steps,
        dt,
        spiral_radius_mm=spiral_radius_mm,
        spiral_height_mm=spiral_height_mm,
        spiral_omega=spiral_omega,
    )
    node_list = model_data["node_list"]
    dist0 = diamond_cable_length(np.zeros(int(model_data["n_full"])), node_list)
    q_weight, r_weight = mpc_weights_for_model(model_data)
    return DiamondExperimentConfig(
        dt=dt,
        tf=float(td[-1]) if td.size else 0.0,
        td=td,
        zd=zd,
        Np=horizon,
        Q=q_weight,
        R=r_weight,
        u_lb=np.full(4 * horizon, -float(control_limit) / 1000.0),
        u_ub=np.full(4 * horizon, float(control_limit) / 1000.0),
        node_list=node_list,
        dist0=dist0,
    )


def mpc_weights_for_model(model_data) -> tuple[np.ndarray, float]:
    if model_data["type"] == "ssm":
        return SSM_Q_WEIGHT, SSM_R_WEIGHT
    return POD_Q_WEIGHT, POD_R_WEIGHT


def load_diamond_model(model_name: str):
    key = model_name.lower()
    if key not in MODEL_DIRS:
        choices = ", ".join(sorted(MODEL_DIRS))
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {choices}")
    model_dir = MODEL_DIRS[key]
    node_list = _as_node_list(loadmat(model_dir / "Node_list.mat")["Node_list"])
    outdof_endeff = np.asarray(loadmat(model_dir / "outdof_endeff.mat")["outdof_endeff"], dtype=int).reshape(-1)
    if key.startswith("pod"):
        phi_r = np.asarray(loadmat(model_dir / "Phi_r.mat")["Phi_r"], dtype=float)
        ar = np.asarray(loadmat(model_dir / "Ar.mat")["Ar"], dtype=float)
        return {
            "type": "pod",
            "model_dir": model_dir,
            "node_list": node_list,
            "outdof_endeff": outdof_endeff,
            "phi_r": phi_r,
            "ar": ar,
            "x_num": ar.shape[0],
            "n_full": phi_r.shape[0],
        }

    n_full = int(np.asarray(loadmat(model_dir / "n.mat")["n"]).reshape(-1)[0])
    return {
        "type": "ssm",
        "model_dir": model_dir,
        "node_list": node_list,
        "outdof_endeff": outdof_endeff,
        "au_data": loadmat(model_dir / "auData.mat")["auData"],
        "w0": loadmat(model_dir / "W_0.mat")["W_0"],
        "x_num": np.asarray(get_field(loadmat(model_dir / "auData.mat")["auData"], "lamd")).reshape(-1).size,
        "n_full": n_full,
    }


def build_controller(model_data, config, factor):
    x_num = int(model_data["x_num"])
    state = DiamondFeedbackState(
        t=0.0,
        k=0,
        x=np.zeros(x_num),
        s0_opt=np.zeros((x_num, config.Np)),
        u_opt=np.zeros((4, config.Np)),
    )
    if model_data["type"] == "pod":
        return DiamondPodMpcController(
            config,
            model_data["ar"],
            model_data["phi_r"],
            model_data["outdof_endeff"],
            state,
            factor=factor,
        )

    w0 = clone_struct_array(model_data["w0"])
    n_full = int(model_data["n_full"])
    out_rows = np.concatenate([model_data["outdof_endeff"], model_data["outdof_endeff"] + n_full])
    cable_rows = cable_dofs_from_node_list(model_data["node_list"])
    return DiamondSsmMpcController(
        config,
        model_data["au_data"],
        w0,
        subset_mapping_rows(w0, out_rows),
        subset_mapping_rows(w0, cable_rows),
        state,
        n_full=n_full,
        factor=factor,
    )


def format_motor_positions(motor_positions) -> str:
    return "[" + ", ".join(f"{value:8.3f}" for value in np.asarray(motor_positions).reshape(-1)) + "]"


def print_timing_summary(records):
    if not records:
        print("\nTiming summary: no completed steps.")
        return
    total = np.asarray([item["total"] for item in records], dtype=float)
    measure = np.asarray([item.get("measure", 0.0) for item in records], dtype=float)
    mpc = np.asarray([item["mpc"] for item in records], dtype=float)
    print("\nTiming summary")
    print(f"  completed steps : {len(records)}")
    print(f"  avg total       : {np.mean(total):.6f}s")
    print(f"  avg measure     : {np.mean(measure):.6f}s")
    print(f"  avg mpc         : {np.mean(mpc):.6f}s")
    print(f"  max mpc         : {np.max(mpc):.6f}s")


def connect_sdk():
    sdk = load_lumo_sdk(BASE_DIR)
    sdk.Init()
    if not sdk.Connect(IP_ADDRESS):
        raise ConnectionError(f"Failed to connect LuMo SDK at {IP_ADDRESS}")
    print("LuMo SDK connected successfully")
    return sdk


@fixed_cpu_time(CONTROL_PERIOD_SECONDS)
def run_step(controller, socket, send_enabled, sdk=None, reference=None, measurement_enabled=False):
    measure_time = 0.0
    measurement = np.zeros(9)
    if measurement_enabled:
        if sdk is None or reference is None:
            raise ValueError("sdk and reference are required when measurement_enabled=True")
        from measure.motion_capture import capture_motion

        measure_start = time.perf_counter()
        disp_end, disp_mid, disp_checkpoint = capture_motion(
            sdk,
            reference,
            mid_rigid_id=MID_RIGID_ID,
            checkpoint_rigid_id=CHECKPOINT_RIGID_ID,
            end_marker_id=END_MARKER_ID,
        )
        measure_time = time.perf_counter() - measure_start
        measurement = np.concatenate([disp_end, disp_mid, disp_checkpoint])

    mpc_start = time.perf_counter()
    motor_positions = controller.compute_next_command()
    mpc_time = time.perf_counter() - mpc_start
    if send_enabled:
        send_motor_command(socket, motor_positions.tolist(), solver_order=False)
    return motor_positions, measurement, measure_time, mpc_time


def main(args):
    original_stdout = sys.stdout
    logger = BufferedTeeLogger(original_stdout)
    sys.stdout = logger
    sdk = None
    reference = None
    context = None
    socket = None
    records = []
    try:
        model_data = load_diamond_model(args.model)
        config = build_config(
            model_data,
            args.trajectory,
            args.steps,
            args.dt,
            args.horizon,
            args.spiral_radius_mm,
            args.spiral_height_mm,
            args.spiral_omega,
            args.control_limit,
        )
        controller = build_controller(model_data, config, factor=args.motor_factor)
        if args.measurement_enabled:
            from measure.motion_capture import initialize_motion_reference

            sdk = connect_sdk()
            reference = initialize_motion_reference(
                sdk,
                base_rigid_id=BASE_RIGID_ID,
                mid_rigid_id=MID_RIGID_ID,
                checkpoint_rigid_id=CHECKPOINT_RIGID_ID,
                end_marker_id=END_MARKER_ID,
            )
        if args.send:
            context, socket = create_publisher(args.endpoint)

        print(f"Using Diamond model : {args.model.lower()} ({model_data['model_dir']})")
        trajectory_name = normalize_trajectory_name(args.trajectory)
        print(f"Trajectory          : {trajectory_name}, steps={config.td.size}, dt={args.dt:g}s")
        print(f"Control mode        : {'ZMQ send enabled' if args.send else 'offline simulation'}")
        print(f"Motion capture      : {'enabled' if args.measurement_enabled else 'disabled'}")
        print(f"Initial cable length: {format_motor_positions(config.dist0 * 1000.0)} mm")

        step_count = int(config.td.size)
        y_expm = np.zeros((step_count, 9))
        y_model = np.zeros((step_count, 6))
        x_save = np.zeros((step_count, model_data["x_num"]))
        u_save = np.zeros((step_count, 4))
        max_mpc_time = 0.0
        for step_index in range(step_count):
            start = time.perf_counter()
            motor_positions, measurement, measure_time, mpc_time = run_step(
                controller,
                socket,
                args.send,
                sdk=sdk,
                reference=reference,
                measurement_enabled=args.measurement_enabled,
            )

            y_expm[step_index, :] = measurement
            y_model[step_index, :] = controller.yk_model * 1000.0
            x_save[step_index, :] = controller.feedback_state.x
            u_save[step_index, :] = motor_positions
            total_time = time.perf_counter() - start
            max_mpc_time = max(max_mpc_time, mpc_time)
            records.append({"total": total_time, "measure": measure_time, "mpc": mpc_time})
            print(
                f"Step {step_index + 1:04d}/{step_count:04d} | "
                f"total {total_time:8.4f}s | "
                f"measure {measure_time:8.4f}s | "
                f"mpc {mpc_time:8.4f}s | "
                f"mpc max {max_mpc_time:8.4f}s | "
                f"motor {format_motor_positions(motor_positions)}"
            )

        result_dir = RESULT_DIR / args.model.lower() / trajectory_name
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "y_exprm_diamond", y_expm)
        np.save(result_dir / "y_model_diamond", y_model)
        np.save(result_dir / "x_save_diamond", x_save)
        np.save(result_dir / "u_save_diamond", u_save)
        np.save(result_dir / "t_diamond", config.td)
        np.save(result_dir / "zd_diamond", config.zd * 1000.0)
        sciIO.savemat(
            result_dir / "diamond_mpc_results.mat",
            {
                "y_exprm_mm": y_expm,
                "y_model_mm": y_model,
                "x_save": x_save,
                "u_save_mm": u_save,
                "t": config.td,
                "zd_mm": config.zd * 1000.0,
            },
        )
        print(f"\nResults saved to {result_dir}")
    finally:
        print_timing_summary(records)
        if socket is not None and context is not None:
            close_publisher(context, socket)
        if sdk is not None:
            sdk.Close()
        sys.stdout = original_stdout
        log_path = logger.save(LOG_DIR)
        print(f"Log saved to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Diamond Robot MPC with POD6, POD26, or SSM6 models.")
    parser.add_argument("--model", choices=sorted(MODEL_DIRS), default=MODEL, help="Diamond model to use.")
    parser.add_argument("--trajectory", "-t", choices=sorted(TRAJECTORY_ALIASES), default=TRAJECTORY_NAME)
    parser.add_argument("--steps", type=int, default=RUN_STEPS, help="Number of MPC steps to simulate.")
    parser.add_argument("--dt", type=float, default=CONTROL_PERIOD_SECONDS, help="Control period in seconds.")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="MPC prediction horizon.")
    parser.add_argument("--spiral-radius-mm", type=float, default=12.0, help="Spiral3D radius_max parameter in mm.")
    parser.add_argument("--spiral-height-mm", type=float, default=0.0, help="Spiral3D height parameter in mm.")
    parser.add_argument("--spiral-omega", type=float, default=2.0 * np.pi / 4.0, help="Spiral3D omega parameter in rad/s.")
    parser.add_argument("--control-limit", type=float, default=MOTOR_SAFETY_LIMIT, help="Cable command limit in mm.")
    parser.add_argument("--motor-factor", type=float, default=MOTOR_FACTOR, help="Scale factor for motor commands.")
    parser.add_argument("--measurement", dest="measurement_enabled", action="store_true", default=MEASUREMENT_ENABLED)
    parser.add_argument("--no-measurement", dest="measurement_enabled", action="store_false")
    parser.add_argument("--send", action="store_true", help="Send motor commands over ZMQ.")
    parser.add_argument("--endpoint", default=MOTOR_ENDPOINT, help="ZMQ motor endpoint used with --send.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
