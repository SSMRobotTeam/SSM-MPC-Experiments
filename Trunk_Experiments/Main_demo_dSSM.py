import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io as sciIO


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller.nmpc_datassm import MOVE_WEIGHT, DataSsmExperimentConfig, DataSsmMpcController, plan_trajectory
from dynamics.datassm_model import MODEL_PATH, DataSsmModel, load_model, verify_coordinate_contract


REFERENCE_ROOT = REPO_ROOT
TRAJECTORY_NAMES = ('eight', 'spiral')
MPC_HORIZON = 3
Q_WEIGHT = 1e3 * np.eye(3)
QF_WEIGHT = 1e3 * np.eye(3)
R_WEIGHT = np.diag([3e-3] * 3 + [8e-3] * 3)
MOTOR_COUNT = 6


def load_trajectory(name):
    if name not in TRAJECTORY_NAMES:
        raise ValueError(f'Unknown trajectory: {name}')
    folder = REFERENCE_ROOT / 'dynamics' / 'trajinfom' / name
    data = sciIO.loadmat(folder / 'zd_proj.mat')
    candidates = [np.asarray(value, dtype=float) for key, value in data.items() if not key.startswith('__')]
    if len(candidates) != 1:
        raise ValueError('The reference file must contain one trajectory array.')
    desired = candidates[0]
    if desired.ndim != 2:
        raise ValueError('The reference trajectory must be a matrix.')
    if desired.shape[0] != 3:
        desired = desired.T
    times = np.asarray(sciIO.loadmat(folder / 't_list.mat')['t_list'], dtype=float).reshape(-1)
    if len(times) < 2 or desired.shape != (3, len(times)):
        raise ValueError('The reference coordinates and time grid do not agree.')
    dt = float(times[1] - times[0])
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(times)) or dt <= 0 or (not np.allclose(np.diff(times), dt, rtol=0, atol=1e-09)):
        raise ValueError('The reference must be finite and sampled at a fixed positive period.')
    desired = 1000.0 * desired * np.array([1.0, -1.0, 1.0])[:, None]
    return (desired, dt)


def control_variation_metrics(commands):
    changes = np.diff(np.asarray(commands, dtype=float), axis=0)
    return {'input_step_rms_mm': float(np.sqrt(np.mean(changes ** 2))), 'input_step_max_mm': float(np.max(np.abs(changes)))}


def tracking_errors(desired, actual):
    error = np.asarray(actual) - np.asarray(desired)
    return {'xy_rmse_mm': float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))), 'z_rmse_mm': float(np.sqrt(np.mean(error[:, 2] ** 2))), 'rmse_3d_mm': float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))}


def plot_results(path, desired, prediction, commands, dt, measured=None, measurement_times=None, measured_reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = (np.arange(len(desired)) + 1) * dt
    command_t = np.arange(len(commands)) * dt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(desired[:, 0], desired[:, 1], 'k-', label='Reference')
    axes[0, 0].plot(prediction[:, 0], prediction[:, 1], 'r--', label='dSSM')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set(xlabel='x [mm]', ylabel='y [mm]')
    axes[0, 1].plot(t, desired[:, 2], 'k-', label='Reference')
    axes[0, 1].plot(t, prediction[:, 2], 'r--', label='dSSM')
    axes[0, 1].set(xlabel='t [s]', ylabel='z [mm]')
    axes[1, 0].plot(t, np.linalg.norm(prediction - desired, axis=1), 'r-', label='dSSM')
    axes[1, 0].set(xlabel='t [s]', ylabel='Position error [mm]')
    for index in range(MOTOR_COUNT):
        axes[1, 1].step(command_t, commands[:, index], where='post', label=f'u{index + 1}')
    axes[1, 1].set(xlabel='t [s]', ylabel='Motor displacement [mm]')
    if measured is not None and len(measured):
        count = len(measured)
        measured_t = t[:count] if measurement_times is None else np.asarray(measurement_times)
        target = desired[:count] if measured_reference is None else np.asarray(measured_reference)
        axes[0, 0].plot(measured[:, 0], measured[:, 1], 'b-', label='Measured')
        axes[0, 1].plot(measured_t, measured[:, 2], 'b-', label='Measured')
        axes[1, 0].plot(measured_t, np.linalg.norm(measured - target, axis=1), 'b-', label='Measured')
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_default_config(move_weight=MOVE_WEIGHT):
    return DataSsmExperimentConfig(Np=MPC_HORIZON, Q=Q_WEIGHT, Qf=QF_WEIGHT, R=R_WEIGHT, move_weight=move_weight)


def main(argv=None):
    global REFERENCE_ROOT
    parser = argparse.ArgumentParser(description='dSSM-MPC offline simulation.')
    parser.add_argument('--trajectory', choices=TRAJECTORY_NAMES, default='eight')
    parser.add_argument('--reference-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--move-weight', type=float, default=MOVE_WEIGHT, help='Input increment cost weight.')
    args = parser.parse_args(argv)
    REFERENCE_ROOT = args.reference_root.resolve()
    model = load_model()
    verify_coordinate_contract(model)
    desired, dt = load_trajectory(args.trajectory)
    if not np.isclose(dt, model.dt, rtol=0, atol=1e-09):
        raise ValueError('Reference and model sampling periods differ.')
    out = args.output_dir or BASE_DIR / 'PlotFig' / 'dSSM_results' / (args.trajectory + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=True)
    for name in ('dssm_plan.mat', 'summary.json', 'dssm_tracking.png'):
        if (out / name).exists():
            raise FileExistsError('Choose a fresh output directory to preserve earlier results.')
    result = plan_trajectory(model, desired, build_default_config(args.move_weight))
    result.update(zd=desired[:, 1:].T, t=np.arange(1, desired.shape[1]) * dt, Np=MPC_HORIZON, solver='qpOASES', move_weight=args.move_weight, state_definition='eta in R^4', run_type='same-model offline simulation')
    sciIO.savemat(out / 'dssm_plan.mat', result)
    summary = {**tracking_errors(result['zd'], result['y_ssm']), 'steps': len(result['t']), 'horizon': MPC_HORIZON, 'solver': 'qpOASES', 'QP_solves_per_sample': 1, 'move_weight': args.move_weight, **control_variation_metrics(result['ut_save']), 'mean_compute_time_s': float(np.mean(result['mpc_time'])), 'max_compute_time_s': float(np.max(result['mpc_time'])), 'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), 'program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'controller_sha256': hashlib.sha256((REPO_ROOT / 'controller/nmpc_datassm.py').read_bytes()).hexdigest(), 'model_code_sha256': hashlib.sha256((REPO_ROOT / 'dynamics/datassm_model.py').read_bytes()).hexdigest(), 'state_definition': result['state_definition'], 'run_type': result['run_type']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not args.no_plot:
        plot_results(out / 'dssm_tracking.png', result['zd'], result['y_ssm'], result['ut_save'], dt)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
