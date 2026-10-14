from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sciIO


MODEL_PATH = Path(__file__).resolve().parent / 'dSSM' / 'datassm_model.mat'


def polynomial_features(state, exponents):
    return np.prod(np.asarray(state).reshape(1, -1) ** exponents, axis=1)


@dataclass(frozen=True)
class DataSsmModel:
    R_coeffs: np.ndarray
    R_exps: np.ndarray
    Br: np.ndarray
    V_xy_coeffs: np.ndarray
    V_exps: np.ndarray
    Wz_coeffs: np.ndarray
    Wz_exps: np.ndarray
    dt: float
    Msub: int
    factor: float
    sentMin: float
    sentMax: float
    uMinR: float
    uMaxR: float
    SSMDim: int
    nu: int

    def dynamics(self, eta):
        return self.R_coeffs @ polynomial_features(eta, self.R_exps)

    def step(self, eta, u):
        h = self.dt / self.Msub
        state = np.asarray(eta, dtype=float).reshape(self.SSMDim).copy()
        control = self.Br @ np.asarray(u, dtype=float).reshape(self.nu)
        for _ in range(self.Msub):
            k1 = self.dynamics(state) + control
            k2 = self.dynamics(state + h * k1 / 2) + control
            k3 = self.dynamics(state + h * k2 / 2) + control
            k4 = self.dynamics(state + h * k3) + control
            state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state

    def output(self, eta):
        xy = self.V_xy_coeffs @ polynomial_features(eta, self.V_exps)
        z = self.Wz_coeffs @ polynomial_features(eta, self.Wz_exps)
        return np.concatenate([xy, z])


def load_model(path=MODEL_PATH):
    data = sciIO.loadmat(path)
    arrays = {name: np.asarray(data[name], dtype=float) for name in ('R_coeffs', 'Br', 'V_xy_coeffs', 'Wz_coeffs')}
    for name in ('R_exps', 'V_exps', 'Wz_exps'):
        values = np.asarray(data[name], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f'Invalid polynomial exponents: {name}')
        arrays[name] = values.astype(np.int64)
    scalars = {name: float(np.asarray(data[name]).item()) for name in ('dt', 'factor', 'sentMin', 'sentMax', 'uMinR', 'uMaxR')}
    integers = {name: int(np.asarray(data[name]).item()) for name in ('Msub', 'SSMDim', 'nu')}
    model = DataSsmModel(**arrays, **scalars, **integers)
    if not all((np.all(np.isfinite(value)) for value in arrays.values())):
        raise ValueError('Model coefficients must be finite.')
    if not all((np.isfinite(value) for value in scalars.values())):
        raise ValueError('Model settings must be finite.')
    if model.SSMDim != 4 or model.nu != 6 or model.dt <= 0 or (model.Msub < 1) or (model.factor <= 0):
        raise ValueError('Invalid Trunk Robot model dimensions or sampling settings.')
    for coeffs, exps, rows in ((model.R_coeffs, model.R_exps, model.SSMDim), (model.V_xy_coeffs, model.V_exps, 2), (model.Wz_coeffs, model.Wz_exps, 1)):
        if exps.ndim != 2 or exps.shape[1] != model.SSMDim or coeffs.shape != (rows, exps.shape[0]):
            raise ValueError('Model polynomial dimensions do not agree.')
    if model.Br.shape != (model.SSMDim, model.nu):
        raise ValueError('Invalid input matrix dimensions.')
    if model.uMinR >= model.uMaxR or model.sentMin >= model.sentMax:
        raise ValueError('Invalid control bounds.')
    return model


def verify_coordinate_contract(model):
    if model.SSMDim != 4 or model.nu != 6:
        raise ValueError('This entry requires four xy/velocity states and six inputs.')
    exps = model.R_exps
    linear = np.eye(4, dtype=int)
    idx = [np.flatnonzero(np.all(exps == p, axis=1)) for p in linear]
    if any((len(i) != 1 for i in idx)):
        raise ValueError('The dynamics must contain one of each linear monomial.')
    expected = np.zeros_like(model.R_coeffs[:2])
    expected[0, idx[2][0]] = 1
    expected[1, idx[3][0]] = 1
    if not np.array_equal(model.R_coeffs[:2], expected) or np.any(model.Br[:2] != 0):
        raise ValueError('Position_dot = velocity is not satisfied.')
    for coeffs, powers in ((model.R_coeffs, model.R_exps), (model.V_xy_coeffs, model.V_exps), (model.Wz_coeffs, model.Wz_exps)):
        if np.any(powers.sum(axis=1) < 1) or np.any(powers.sum(axis=1) > 3):
            raise ValueError('Expected degree one to three without constant terms.')
    expected_xy = np.zeros_like(model.V_xy_coeffs)
    for row in range(2):
        positions = np.flatnonzero(np.all(model.V_exps == linear[row], axis=1))
        if len(positions) != 1:
            raise ValueError('Missing position output monomial.')
        expected_xy[row, positions[0]] = 1
    if not np.array_equal(model.V_xy_coeffs, expected_xy):
        raise ValueError('The first two outputs must equal the position states.')
    velocity_terms = np.any(model.Wz_exps[:, 2:] != 0, axis=1)
    if np.any(model.Wz_coeffs[:, velocity_terms] != 0):
        raise ValueError('The supplied height map is not position-only.')
