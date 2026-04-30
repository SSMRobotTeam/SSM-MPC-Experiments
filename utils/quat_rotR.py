import numpy as np


def quat2dcm(quat, normalize: bool = True) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,) in [qw, qx, qy, qz] order")

    if normalize:
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            raise ValueError("Quaternion norm is zero; cannot normalize")
        q = q / norm

    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y + 2 * z * w, 2 * x * z - 2 * y * w],
            [2 * x * y - 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z + 2 * x * w],
            [2 * x * z + 2 * y * w, 2 * y * z - 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
