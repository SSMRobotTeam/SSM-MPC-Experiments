import numpy as np


def c_curve(t, T):
    t = np.asarray(t, dtype=float).reshape(-1)
    x_u = t / T * 4.0 - 2.0
    y_u = -(x_u**2) + 1.0

    zd = np.zeros((3, t.size))
    zd[1, :] = (x_u + 2.0) * 25.0 / 1000.0
    zd[0, :] = -(y_u + 3.0) * 25.0 / 1000.0
    return zd


def eight_curve(t, T, A, outdof_endeff):
    t = np.asarray(t, dtype=float).reshape(-1)
    theta = (t - 0.25 * T) / T * (2.0 * np.pi)
    zd = np.zeros((len(outdof_endeff), t.size))
    zd[0, :] = 0.35 * A * np.cos(theta) / (1.0 + np.sin(theta) ** 2)
    zd[1, :] = A * np.cos(theta) * np.sin(theta) / (1.0 + np.sin(theta) ** 2)
    return zd


def function_traj_dtraj(t, T1, Teight, P_start, dP_start, P_end, dP_end, A, omega, name):
    t = np.asarray(t, dtype=float).reshape(-1)
    P_start = np.asarray(P_start, dtype=float).reshape(3)
    dP_start = np.asarray(dP_start, dtype=float).reshape(3)
    P_end = np.asarray(P_end, dtype=float).reshape(3)
    dP_end = np.asarray(dP_end, dtype=float).reshape(3)

    c0 = P_start
    c1 = P_start + T1 * dP_start / 3.0
    c2 = P_end - T1 * dP_end / 3.0
    c3 = P_end
    coeffs = np.vstack([c0, c1, c2, c3])

    zout = np.zeros((3, t.size))
    first_after = np.flatnonzero(t > T1)
    first_index = t.size if first_after.size == 0 else int(first_after[0])

    t1 = t[:first_index]
    tau = t1 / T1
    if name == "Traj":
        basis = np.vstack(
            [
                (1 - tau) ** 3,
                3 * tau * (1 - tau) ** 2,
                3 * tau**2 * (1 - tau),
                tau**3,
            ]
        )
        zout[:, :first_index] = (basis.T @ coeffs).T
    elif name == "dTraj":
        basis = np.vstack(
            [
                -3 / T1 * (1 - tau) ** 2,
                3 / T1 * (1 - tau) * (1 - 3 * tau),
                (3 * t1 / T1**2) * (2 - 3 * tau),
                (3 / T1) * tau**2,
            ]
        )
        zout[:, :first_index] = (basis.T @ coeffs).T
    else:
        raise ValueError("name must be 'Traj' or 'dTraj'")

    t2 = t[first_index:] - T1
    theta = (t2 - 0.25 * Teight) / Teight * (2 * np.pi)
    if t2.size and name == "Traj":
        zout[:, first_index:] = np.vstack(
            [
                P_end[0] + np.zeros_like(theta),
                P_end[1] + 0.35 * A * np.cos(theta) / (1 + np.sin(theta) ** 2),
                P_end[2] + A * np.cos(theta) * np.sin(theta) / (1 + np.sin(theta) ** 2),
            ]
        )
    elif t2.size:
        zout[:, first_index:] = np.vstack(
            [
                -0.35 * A * omega * np.sin(theta) * (3 - np.sin(theta) ** 2) / (1 + np.sin(theta) ** 2) ** 2,
                np.zeros_like(theta),
                A * omega * (1 - 3 * np.sin(theta) ** 2) / (1 + np.sin(theta) ** 2) ** 2,
            ]
        )
    return zout

