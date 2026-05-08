from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


Array = np.ndarray


def hat(w: Sequence[float]) -> Array:
    wx, wy, wz = np.asarray(w, dtype=float)
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
        dtype=float,
    )


def vee(W: Array) -> Array:
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


def exp_so3(w: Sequence[float], length: float) -> Array:
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w) * length
    W = hat(w)
    if theta < 1e-10:
        return np.eye(3) + W * length + 0.5 * (W @ W) * length * length
    wn = np.linalg.norm(w)
    return (
        np.eye(3)
        + np.sin(theta) / wn * W
        + (1.0 - np.cos(theta)) / (wn * wn) * (W @ W)
    )


def split_constant_strain(strain: Sequence[float]) -> Tuple[Array, Array]:
    """Return angular and translational strain vectors for a PCS unit.

    Supported strain layouts are:
        4D: [kappa_x, kappa_y, tau, epsilon]
        6D: [kappa_x, kappa_y, tau, epsilon, gamma_x, gamma_y]
    """

    q = np.asarray(strain, dtype=float)
    if q.shape != (4,) and q.shape != (6,):
        raise ValueError(
            "strain must have shape (4,) [kx, ky, tau, eps] or "
            "(6,) [kx, ky, tau, eps, gamma_x, gamma_y]"
        )
    if q.size == 4:
        kappa_x, kappa_y, tau, epsilon = q
        gamma_x = 0.0
        gamma_y = 0.0
    else:
        kappa_x, kappa_y, tau, epsilon, gamma_x, gamma_y = q
    w = np.array([kappa_x, kappa_y, tau], dtype=float)
    v = np.array([gamma_x, gamma_y, 1.0 + epsilon], dtype=float)
    return w, v


def exp_se3_constant_strain(strain: Sequence[float], length: float) -> Array:
    """Homogeneous transform of a constant-strain Cosserat segment.

    The 4D layout keeps the historical shear-free model. The 6D layout includes
    shear strains gamma_x and gamma_y in the local translational strain vector.
    """

    w, v = split_constant_strain(strain)

    W = hat(w)
    wn = np.linalg.norm(w)
    R = exp_so3(w, length)
    if wn < 1e-10:
        p = v * length + 0.5 * (W @ v) * length * length
    else:
        p = (
            np.eye(3) * length
            + (1.0 - np.cos(wn * length)) / (wn * wn) * W
            + (wn * length - np.sin(wn * length)) / (wn**3) * (W @ W)
        ) @ v

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


@dataclass(frozen=True)
class Tendon:
    """Tendon routing in one section cross-section.

    angle is measured in the local x-y plane. radius can be a constant or
    (r_start, r_end). For the simplified slow-r(s) model the mean radius is used.
    """

    angle: float
    radius: float | Tuple[float, float]

    @property
    def mean_radius(self) -> float:
        if isinstance(self.radius, tuple):
            return 0.5 * (float(self.radius[0]) + float(self.radius[1]))
        return float(self.radius)


@dataclass(frozen=True)
class PCSSection:
    length: float
    tendons: Tuple[Tendon, ...]

    @classmethod
    def with_uniform_tendons(
        cls,
        length: float,
        radius: float | Tuple[float, float],
        angles: Iterable[float],
    ) -> "PCSSection":
        return cls(length=float(length), tendons=tuple(Tendon(a, radius) for a in angles))


def _as_section_strains(
    strains: Sequence[Sequence[float]],
    section_count: Optional[int] = None,
    strain_size: Optional[int] = None,
) -> Array:
    arr = np.asarray(strains, dtype=float)
    valid_sizes = (4, 6) if strain_size is None else (int(strain_size),)
    if arr.ndim != 2 or arr.shape[1] not in valid_sizes:
        if strain_size is None:
            raise ValueError("strains must have shape (N, 4) or (N, 6).")
        raise ValueError(f"strains must have shape (N, {strain_size}), got {arr.shape}")
    if section_count is not None and arr.shape[0] != section_count:
        expected_cols = arr.shape[1] if strain_size is None else strain_size
        raise ValueError(f"strains must have shape ({section_count}, {expected_cols}), got {arr.shape}")
    return arr


def pose_error(current: Array, target: Array) -> Array:
    """Small-angle SE(3) pose error [dx, dy, dz, rx, ry, rz]."""

    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    R_err = target[:3, :3] @ current[:3, :3].T
    rot = 0.5 * vee(R_err - R_err.T)
    trans = target[:3, 3] - current[:3, 3]
    return np.concatenate([trans, rot])


def tendon_radius_at(tendon: Tendon, s: float, length: float) -> float:
    if isinstance(tendon.radius, tuple):
        r0, r1 = tendon.radius
        return float(r0 + (r1 - r0) * s / length)
    return float(tendon.radius)


class TwoSectionPCSRobot:
    """Kinematic control model for a two-section trunk discretized into PCS units."""

    def __init__(
        self,
        sections: Sequence[PCSSection],
        short_tendons: Optional[Sequence[Tendon]] = None,
        short_section_count: Optional[int] = None,
        strain_size: int = 4,
    ):
        if len(sections) < 2:
            raise ValueError("TwoSectionPCSRobot requires at least two PCS units.")
        if strain_size not in (4, 6):
            raise ValueError("strain_size must be 4 for the simplified model or 6 for the full shear model.")
        self.sections = tuple(sections)
        self.short_tendons = tuple(short_tendons) if short_tendons is not None else self.sections[0].tendons
        self.short_section_count = len(self.sections) // 2 if short_section_count is None else int(short_section_count)
        self.strain_size = int(strain_size)
        if self.short_section_count < 1 or self.short_section_count > len(self.sections):
            raise ValueError("short_section_count must be between 1 and the PCS unit count.")
        if len(self.short_tendons) != len(self.sections[0].tendons):
            raise ValueError("short_tendons must match the long tendon count.")
        for section in self.sections[1:]:
            if len(section.tendons) != len(self.sections[0].tendons):
                raise ValueError("All PCS units must use the same long tendon count.")

    def forward_kinematics(self, strains: Sequence[Sequence[float]]) -> List[Array]:
        q = _as_section_strains(strains, len(self.sections), self.strain_size)
        transforms = []
        T = np.eye(4)
        for strain, section in zip(q, self.sections):
            T = T @ exp_se3_constant_strain(strain, section.length)
            transforms.append(T.copy())
        return transforms

    def tip_transform(self, strains: Sequence[Sequence[float]]) -> Array:
        return self.forward_kinematics(strains)[-1]

    def tip_position(self, strains: Sequence[Sequence[float]]) -> Array:
        return self.tip_transform(strains)[:3, 3].copy()

    def backbone_points(
        self,
        strains: Sequence[Sequence[float]],
        samples_per_section: int = 40,
    ) -> Array:
        """Sample centerline points from base to tip."""

        if samples_per_section < 2:
            raise ValueError("samples_per_section must be at least 2.")

        q = _as_section_strains(strains, len(self.sections), self.strain_size)
        points = [np.zeros(3, dtype=float)]
        base = np.eye(4)
        for section_index, (strain, section) in enumerate(zip(q, self.sections)):
            start = 1 if section_index == 0 else 0
            for s in np.linspace(0.0, section.length, samples_per_section)[start:]:
                T = base @ exp_se3_constant_strain(strain, float(s))
                points.append(T[:3, 3].copy())
            base = base @ exp_se3_constant_strain(strain, section.length)
        return np.asarray(points, dtype=float)

    def section_tendon_lengths(
        self, section_index: int, strain: Sequence[float], simplified: bool = True
    ) -> Array:
        """Return tendon lengths for one PCS section.

        With 4D strains this is the historical shear-free tendon model. With
        6D strains it uses ||v + u x r_i||, so gamma_x and gamma_y contribute
        directly to tendon length. The simplified path neglects dr/ds for slow
        radius variation; the integrated path includes it.
        """

        return self._section_tendon_lengths(self.sections[section_index], self.sections[section_index].tendons, strain, simplified)

    def _section_tendon_lengths(
        self,
        section: PCSSection,
        tendons: Sequence[Tendon],
        strain: Sequence[float],
        simplified: bool = True,
    ) -> Array:
        """Return tendon lengths for a provided tendon set in one PCS section."""

        w, v = split_constant_strain(strain)
        kx, ky, tau = w
        lengths = []

        for tendon in tendons:
            phi = tendon.angle
            r = tendon.mean_radius
            shear_x = v[0] - tau * r * np.sin(phi)
            shear_y = v[1] + tau * r * np.cos(phi)
            axial_term = v[2] + r * (kx * np.sin(phi) - ky * np.cos(phi))
            if simplified:
                integrand = np.sqrt(shear_x**2 + shear_y**2 + axial_term**2)
                lengths.append(section.length * integrand)
            else:
                lengths.append(self._integrated_tendon_length(section, tendon, kx, ky, tau, v))
        return np.asarray(lengths, dtype=float)

    def _integrated_tendon_length(
        self,
        section: PCSSection,
        tendon: Tendon,
        kx: float,
        ky: float,
        tau: float,
        v: Array,
        samples: int = 64,
    ) -> float:
        if not isinstance(tendon.radius, tuple):
            r = tendon.mean_radius
            return section.length * np.sqrt(
                (v[0] - tau * r * np.sin(tendon.angle)) ** 2
                + (v[1] + tau * r * np.cos(tendon.angle)) ** 2
                + (v[2] + r * (kx * np.sin(tendon.angle) - ky * np.cos(tendon.angle))) ** 2
            )
        s = np.linspace(0.0, section.length, samples)
        r0, r1 = tendon.radius
        r = r0 + (r1 - r0) * s / section.length
        dr_ds = (r1 - r0) / section.length
        radial_x = dr_ds * np.cos(tendon.angle)
        radial_y = dr_ds * np.sin(tendon.angle)
        shear_x = v[0] - tau * r * np.sin(tendon.angle) + radial_x
        shear_y = v[1] + tau * r * np.cos(tendon.angle) + radial_y
        axial = v[2] + r * (kx * np.sin(tendon.angle) - ky * np.cos(tendon.angle))
        integrand = np.sqrt(shear_x**2 + shear_y**2 + axial**2)
        return float(np.trapezoid(integrand, s))

    def tendon_lengths(self, strains: Sequence[Sequence[float]], simplified: bool = True) -> Array:
        """Return six routed actuator lengths.

        For three tendons per section the output order is:
        [long_1, long_2, long_3, short_1, short_2, short_3].
        Long tendons pass through both sections, short tendons only through the
        proximal section.
        """

        q = _as_section_strains(strains, len(self.sections), self.strain_size)
        long_tendons = np.zeros(len(self.sections[0].tendons), dtype=float)
        for section_index, strain in enumerate(q):
            long_tendons += self.section_tendon_lengths(section_index, strain, simplified=simplified)

        short_tendons = np.zeros(len(self.short_tendons), dtype=float)
        for section_index in range(self.short_section_count):
            short_tendons += self._section_tendon_lengths(
                self.sections[section_index],
                self.short_tendons,
                q[section_index],
                simplified=simplified,
            )
        return np.concatenate([long_tendons, short_tendons])

    def tendon_length_names(self) -> List[str]:
        count = len(self.sections[0].tendons)
        return [f"long_{i + 1}" for i in range(count)] + [f"short_{i + 1}" for i in range(count)]

    def routed_tendon_paths(
        self,
        strains: Sequence[Sequence[float]],
        samples_per_section: int = 40,
    ) -> dict[str, List[Array]]:
        """Sample spatial paths of routed long and short tendons."""

        if samples_per_section < 2:
            raise ValueError("samples_per_section must be at least 2.")

        q = _as_section_strains(strains, len(self.sections), self.strain_size)
        paths: dict[str, List[Array]] = {"long": [], "short": []}

        base_transforms = [np.eye(4)]
        base = np.eye(4)
        for strain, section in zip(q, self.sections):
            base = base @ exp_se3_constant_strain(strain, section.length)
            base_transforms.append(base.copy())

        for tendon_index in range(len(self.sections[0].tendons)):
            points = []
            for section_index, section in enumerate(self.sections):
                tendon = section.tendons[tendon_index]
                s_values = np.linspace(0.0, section.length, samples_per_section)
                if section_index > 0:
                    s_values = s_values[1:]
                for s in s_values:
                    T = base_transforms[section_index] @ exp_se3_constant_strain(q[section_index], float(s))
                    r = tendon_radius_at(tendon, float(s), section.length)
                    local = np.array([r * np.cos(tendon.angle), r * np.sin(tendon.angle), 0.0])
                    points.append(T[:3, :3] @ local + T[:3, 3])
            paths["long"].append(np.asarray(points, dtype=float))

        for tendon in self.short_tendons:
            points = []
            for section_index in range(self.short_section_count):
                section = self.sections[section_index]
                s_values = np.linspace(0.0, section.length, samples_per_section)
                if section_index > 0:
                    s_values = s_values[1:]
                for s in s_values:
                    T = base_transforms[section_index] @ exp_se3_constant_strain(q[section_index], float(s))
                    r = tendon_radius_at(tendon, float(s), section.length)
                    local = np.array([r * np.cos(tendon.angle), r * np.sin(tendon.angle), 0.0])
                    points.append(T[:3, :3] @ local + T[:3, 3])
            paths["short"].append(np.asarray(points, dtype=float))

        return paths

    def tendon_jacobian(self, strains: Sequence[Sequence[float]], step: float = 1e-6) -> Array:
        q = _as_section_strains(strains, len(self.sections), self.strain_size)
        base = self.tendon_lengths(q)
        J = np.zeros((base.size, q.size), dtype=float)
        for col in range(q.size):
            perturbed = q.copy().reshape(-1)
            perturbed[col] += step
            lengths = self.tendon_lengths(perturbed.reshape(q.shape))
            J[:, col] = (lengths - base) / step
        return J

    def task_jacobian(self, strains: Sequence[Sequence[float]], step: float = 1e-6) -> Array:
        q = _as_section_strains(strains, len(self.sections), self.strain_size)
        base_T = self.tip_transform(q)
        J = np.zeros((6, q.size), dtype=float)
        for col in range(q.size):
            perturbed = q.copy().reshape(-1)
            perturbed[col] += step
            T = self.tip_transform(perturbed.reshape(q.shape))
            J[:, col] = pose_error(base_T, T) / step
        return J

    def position_jacobian(self, strains: Sequence[Sequence[float]], step: float = 1e-6) -> Array:
        q = _as_section_strains(strains, len(self.sections), self.strain_size)
        base_position = self.tip_position(q)
        J = np.zeros((3, q.size), dtype=float)
        for col in range(q.size):
            perturbed = q.copy().reshape(-1)
            perturbed[col] += step
            position = self.tip_position(perturbed.reshape(q.shape))
            J[:, col] = (position - base_position) / step
        return J

    def solve_strains_from_tendon_lengths(
        self,
        target_lengths: Sequence[float],
        initial_strains: Optional[Sequence[Sequence[float]]] = None,
        damping: float = 1e-6,
        max_iter: int = 80,
        tol: float = 1e-9,
    ) -> Array:
        target = np.asarray(target_lengths, dtype=float)
        q = (
            np.zeros((len(self.sections), self.strain_size), dtype=float)
            if initial_strains is None
            else _as_section_strains(initial_strains, len(self.sections), self.strain_size).copy()
        )
        if target.shape != self.tendon_lengths(q).shape:
            raise ValueError("target_lengths size must match the total tendon count.")

        for _ in range(max_iter):
            residual = target - self.tendon_lengths(q)
            if np.linalg.norm(residual) < tol:
                break
            J = self.tendon_jacobian(q)
            A = J.T @ J + damping * np.eye(J.shape[1])
            dq = np.linalg.solve(A, J.T @ residual)
            q += dq.reshape(q.shape)
            if np.linalg.norm(dq) < tol:
                break
        return q

    def solve_strains_to_tip_position_least_squares(
        self,
        target_tip_position: Sequence[float],
        initial_strains: Optional[Sequence[Sequence[float]]] = None,
        R_work: float = 1e3,
        R_strain: float = 1e-2,
        R_smooth: float = 1e-2,
        R_deformation: float = 0.0,
        R_bend: float = 1.0,
        R_twist: float = 1.0,
        R_axial: float = 1.0,
        R_shear: float = 1.0,
        max_nfev: int = 300,
    ):
        """Solve PCS strains for a target tip position using scipy least_squares.

        The objective tracks the target tip position while lightly regularizing
        against the previous strain state, absolute deformation, and
        discontinuities between adjacent PCS units. This keeps the
        underdetermined 3D position IK problem from drifting into unnecessarily
        large or jagged strain states.
        """

        target = np.asarray(target_tip_position, dtype=float).reshape(3)
        q_init = (
            np.zeros((len(self.sections), self.strain_size), dtype=float)
            if initial_strains is None
            else _as_section_strains(initial_strains, len(self.sections), self.strain_size).copy()
        )
        component_weights = self._least_squares_component_weights(
            R_bend=R_bend,
            R_twist=R_twist,
            R_axial=R_axial,
            R_shear=R_shear,
        )

        def residuals(q_flat: Array) -> Array:
            q = q_flat.reshape(q_init.shape)
            r_position = R_work * (self.tip_position(q) - target)
            r_strain = R_strain * ((q - q_init) * component_weights).reshape(-1)
            r_deformation = R_deformation * (q * component_weights).reshape(-1)
            r_smooth = R_smooth * (np.diff(q, axis=0) * component_weights).reshape(-1)
            return np.hstack([r_position, r_strain, r_deformation, r_smooth])
# method="trf", "lm"
        result = least_squares(
            residuals,
            q_init.reshape(-1),
            method="lm",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )
        return result.x.reshape(q_init.shape), result

    def _least_squares_component_weights(
        self,
        R_bend: float,
        R_twist: float,
        R_axial: float,
        R_shear: float,
    ) -> Array:
        if self.strain_size == 4:
            return np.array([R_bend, R_bend, R_twist, R_axial], dtype=float)
        return np.array([R_bend, R_bend, R_twist, R_axial, R_shear, R_shear], dtype=float)

    def control_step_to_tip_pose(
        self,
        strains: Sequence[Sequence[float]],
        target_tip_transform: Array,
        gain: float = 0.4,
        damping: float = 1e-5,
        max_delta_norm: float = 0.1,
    ) -> Tuple[Array, Array]:
        """One resolved-rate control step.

        Returns (next_strains, next_tendon_lengths). The tendon lengths are the
        actuator commands corresponding to the updated PCS generalized strain.
        """

        q = _as_section_strains(strains, len(self.sections), self.strain_size).copy()
        err = pose_error(self.tip_transform(q), target_tip_transform)
        J = self.task_jacobian(q)
        A = J.T @ J + damping * np.eye(J.shape[1])
        dq = gain * np.linalg.solve(A, J.T @ err)
        norm = np.linalg.norm(dq)
        if norm > max_delta_norm:
            dq *= max_delta_norm / norm
        next_q = q + dq.reshape(q.shape)
        return next_q, self.tendon_lengths(next_q)

    def control_step_to_tip_position(
        self,
        strains: Sequence[Sequence[float]],
        target_tip_position: Sequence[float],
        gain: float = 0.4,
        damping: float = 1e-5,
        max_delta_norm: float = 0.1,
    ) -> Tuple[Array, Array]:
        q = _as_section_strains(strains, len(self.sections), self.strain_size).copy()
        target = np.asarray(target_tip_position, dtype=float)
        if target.shape != (3,):
            raise ValueError("target_tip_position must have shape (3,).")
        err = target - self.tip_position(q)
        J = self.position_jacobian(q)
        A = J.T @ J + damping * np.eye(J.shape[1])
        dq = gain * np.linalg.solve(A, J.T @ err)
        norm = np.linalg.norm(dq)
        if norm > max_delta_norm:
            dq *= max_delta_norm / norm
        next_q = q + dq.reshape(q.shape)
        return next_q, self.tendon_lengths(next_q)
