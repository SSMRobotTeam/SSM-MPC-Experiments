import numpy as np

from dynamics.SSM.matlab_data import get_field, matlab_struct_array, to_dense


def _ri_state(z):
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    z_real = np.zeros_like(z, dtype=float)
    z_imag = np.zeros_like(z, dtype=float)
    z_real[0::2, :] = z[0::2, :]
    z_real[1::2, :] = z[0::2, :]
    z_imag[0::2, :] = z[1::2, :]
    z_imag[1::2, :] = -z[1::2, :]
    return z_real, z_imag


def _complex_power(real_part, imag_part, exponent):
    if exponent < 0:
        return np.zeros_like(real_part), np.zeros_like(imag_part)
    value = (real_part + 1j * imag_part) ** int(exponent)
    return np.real(value), np.imag(value)


def _monomial_ri(p_real, p_imag, powers):
    y_real = np.ones((p_real.shape[1],), dtype=float)
    y_imag = np.zeros((p_real.shape[1],), dtype=float)

    for row, power in enumerate(np.asarray(powers).reshape(-1)):
        a2, b2 = _complex_power(p_real[row, :], p_imag[row, :], power)
        y_real, y_imag = y_real * a2 - y_imag * b2, y_real * b2 + a2 * y_imag
    return y_real, y_imag


def expand_multiindex(mapping, p):
    """Python equivalent of MATLAB R_expand_multiindex for numeric inputs."""
    ind = np.asarray(to_dense(get_field(mapping, "ind")))
    if ind.size == 0:
        return 0.0

    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    coeffs = np.asarray(to_dense(get_field(mapping, "coeffs")))
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(-1, 1)
    powers = np.asarray(ind, dtype=int)
    if powers.ndim == 1:
        powers = powers.reshape(1, -1)
    p_real, p_imag = _ri_state(p)

    monomial_real = []
    monomial_imag = []
    for power_row in powers:
        r, im = _monomial_ri(p_real, p_imag, power_row)
        monomial_real.append(r)
        monomial_imag.append(im)

    monomial_real = np.vstack(monomial_real)
    monomial_imag = np.vstack(monomial_imag)
    return np.real(coeffs) @ monomial_real - np.imag(coeffs) @ monomial_imag


def reduced_to_full(p, w0, w1=None, epsilon=1):
    """Evaluate the SSM mapping W(p). Non-autonomous W1 terms are omitted unless provided."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    maps = matlab_struct_array(w0)
    n_rows = np.asarray(get_field(maps[0], "coeffs")).shape[0]
    z = np.zeros((n_rows, p.shape[1]), dtype=float)

    if w1 is not None and epsilon != 0:
        raise NotImplementedError("Non-autonomous W1 terms from MATLAB were not used in this experiment.")

    for mapping in maps:
        z += np.real(expand_multiindex(mapping, p))
    return z


def auto_red_dyn(z, data):
    """Reduced autonomous SSM vector field, ported from R_auto_red_dyn.m."""
    z = np.asarray(z, dtype=float)
    was_vector = z.ndim == 1
    if was_vector:
        z = z.reshape(-1, 1)

    lamd = np.asarray(to_dense(get_field(data, "lamd"))).reshape(-1, 1)
    beta = np.asarray(to_dense(get_field(data, "beta")))
    kappa = np.asarray(to_dense(get_field(data, "kappa")), dtype=int)

    p_real, p_imag = _ri_state(z)
    lamd_real = np.real(lamd)
    lamd_imag = np.imag(lamd)

    y_real = p_real * lamd_real - p_imag * lamd_imag
    y_imag = p_real * lamd_imag + p_imag * lamd_real

    for term_idx, powers in enumerate(kappa):
        coeffs = beta[:, term_idx].reshape(-1, 1)
        if np.max(np.abs(coeffs)) <= 1e-8:
            continue
        term_real, term_imag = _monomial_ri(p_real, p_imag, powers)
        term_real = term_real.reshape(1, -1)
        term_imag = term_imag.reshape(1, -1)
        y_real += np.real(coeffs) @ term_real - np.imag(coeffs) @ term_imag
        y_imag += np.real(coeffs) @ term_imag + np.imag(coeffs) @ term_real

    y = np.zeros_like(z, dtype=float)
    y[0::2, :] = y_real[0::2, :]
    y[1::2, :] = y_imag[0::2, :]
    return y.reshape(-1) if was_vector else y


def jacobian_auto_red_dyn(audata, p0, r_pk0=None, delta=np.finfo(float).eps):
    p0 = np.asarray(p0, dtype=float).reshape(-1)
    if r_pk0 is None:
        r_pk0 = auto_red_dyn(p0, audata)

    perturbed = p0[:, None] + delta * np.eye(p0.size)
    r_perturbed = auto_red_dyn(perturbed, audata)
    return (r_perturbed - np.asarray(r_pk0, dtype=float).reshape(-1, 1)) / delta


def jacobian_w_map(w_map, p0, output_size=None, w_pk0=None, delta=np.finfo(float).eps):
    p0 = np.asarray(p0, dtype=float).reshape(-1)
    if w_pk0 is None:
        w_pk0 = reduced_to_full(p0, w_map, None, 0).reshape(-1)
    if output_size is None:
        output_size = w_pk0.size

    perturbed = p0[:, None] + delta * np.eye(p0.size)
    w_perturbed = reduced_to_full(perturbed, w_map, None, 0)
    return (w_perturbed[:output_size, :] - np.asarray(w_pk0[:output_size], dtype=float).reshape(-1, 1)) / delta
