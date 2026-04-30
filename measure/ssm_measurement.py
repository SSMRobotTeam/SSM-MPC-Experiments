from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sciIO


@dataclass
class SsmMeasurementState:
    t_i: float = 0.0
    displ_endeff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    displ_mideff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    displ_feedback_pt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    y_exprm: list[np.ndarray] = field(default_factory=list)


def initialize_dynamic_capture() -> SsmMeasurementState:
    return SsmMeasurementState()


def append_measurement(state: SsmMeasurementState, displ_endeff, displ_mideff, displ_feedback_pt=None):
    state.displ_endeff = np.asarray(displ_endeff, dtype=float).reshape(3)
    state.displ_mideff = np.asarray(displ_mideff, dtype=float).reshape(3)
    if displ_feedback_pt is None:
        displ_feedback_pt = np.zeros(3)
    state.displ_feedback_pt = np.asarray(displ_feedback_pt, dtype=float).reshape(3)
    state.y_exprm.append(np.concatenate([state.displ_endeff, state.displ_mideff, state.displ_feedback_pt]))
    return state


def save_measurements(state: SsmMeasurementState, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    y_exprm = np.vstack(state.y_exprm) if state.y_exprm else np.empty((0, 9))
    np.save(output_dir / "y_exprm_ssm", y_exprm)
    sciIO.savemat(output_dir / "y_exprm_ssm.mat", {"y_exprm": y_exprm})
    return y_exprm

