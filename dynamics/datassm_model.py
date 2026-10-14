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
    chart_basis: np.ndarray
    W_coeffs: np.ndarray
    W_exps: np.ndarray
    embedding_lags: np.ndarray
    embedding_channels: np.ndarray
    embedding_dt: float

    def encode_history(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (2, 5) or not np.all(np.isfinite(positions)):
            raise ValueError('Expected five chronological xy samples, shape (2, 5).')
        return self.chart_basis.T @ positions.T.reshape(-1)

    def reconstruct_history(self, eta):
        history = self.W_coeffs @ polynomial_features(eta, self.W_exps)
        return history.reshape(5, 2).T

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
    arrays = {name: np.asarray(data[name], dtype=float) for name in ('R_coeffs', 'Br', 'V_xy_coeffs', 'Wz_coeffs', 'chart_basis', 'W_coeffs')}
    for name in ('R_exps', 'V_exps', 'Wz_exps', 'W_exps', 'embedding_lags', 'embedding_channels'):
        values = np.asarray(data[name], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f'Invalid polynomial exponents: {name}')
        arrays[name] = values.astype(np.int64)
    arrays['embedding_lags'] = arrays['embedding_lags'].reshape(-1)
    arrays['embedding_channels'] = arrays['embedding_channels'].reshape(-1)
    scalars = {name: float(np.asarray(data[name]).item()) for name in ('dt', 'factor', 'sentMin', 'sentMax', 'uMinR', 'uMaxR', 'embedding_dt')}
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
    if model.chart_basis.shape != (10, 4):
        raise ValueError('Expected a four-dimensional chart of five xy samples.')
    if not np.array_equal(model.embedding_lags, np.arange(4, -1, -1)):
        raise ValueError('The history must run from k-4 through k.')
    if not np.array_equal(model.embedding_channels, np.array([0, 1])):
        raise ValueError('The chart must use the x and y position channels.')
    if not np.isclose(model.embedding_dt, model.dt, rtol=0, atol=1e-12):
        raise ValueError('The history and dynamics sampling periods differ.')
    if not np.allclose(model.chart_basis.T @ model.chart_basis, np.eye(4), rtol=0, atol=1e-10):
        raise ValueError('The chart basis is not orthonormal.')
    if model.W_coeffs.shape != (10, len(model.W_exps)):
        raise ValueError('Invalid history reconstruction dimensions.')
    if not np.array_equal(model.W_exps, model.V_exps):
        raise ValueError('The history and output polynomial bases differ.')
    if not np.array_equal(model.W_coeffs[-2:], model.V_xy_coeffs):
        raise ValueError('The xy output must extract the current history sample.')
    expected = np.zeros((4, len(model.W_exps)))
    for j, monomial in enumerate(np.eye(4, dtype=int)):
        indices = np.flatnonzero(np.all(model.W_exps == monomial, axis=1))
        if len(indices) != 1:
            raise ValueError('Missing or duplicated chart monomial.')
        expected[j, indices[0]] = 1
    if not np.allclose(model.chart_basis.T @ model.W_coeffs, expected, rtol=0, atol=1e-10):
        raise ValueError('The chart and reconstruction maps are inconsistent.')
    for powers in (model.R_exps, model.V_exps, model.Wz_exps, model.W_exps):
        if powers.ndim != 2 or powers.shape[1] != 4:
            raise ValueError('Expected four-dimensional polynomial exponents.')
        if np.any(powers.sum(axis=1) < 1) or np.any(powers.sum(axis=1) > 3):
            raise ValueError('Expected degree one to three without constant terms.')
