from __future__ import annotations

"""PCS replacement case for continuation-based tension-static IK.

This module mirrors the structure of ``tension_static_optimizer.py`` but replaces
GEF/GEBF static equilibrium with a Piecewise Constant Strain (PCS) Cosserat-rod
static solve. It is intended as a comparison baseline against your geometrically
exact beam model.

Conventions
-----------
- The undeformed rod axis is the local x-axis.
- PCS strain per segment is
      xi_i = [kappa_x, kappa_y, kappa_z, v_x, v_y, v_z]
  where kappa_x is torsion, kappa_y/kappa_z are bending curvatures, v_x is axial
  stretch, and v_y/v_z are shear strains.
- Natural straight strain is [0, 0, 0, 1, 0, 0].
- Cable guide points are fixed material points at prescribed stations s and
  local cross-section offsets [0, r cos(theta), r sin(theta)], matching the
  station-based routing in your single-segment model.
- Cable virtual work is
      delta W_cable = - tau^T delta l(q)
  so Q_cable = -A(q)^T tau, A = dl/dq. The residual is therefore
      r(q) = K(q-q0) + A(q)^T tau - Q_g(q) = 0.
"""

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize

Array = npt.NDArray[np.float64]


# -----------------------------------------------------------------------------
# Options and result containers; names intentionally parallel your GEBF IK layer
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PCSTensionStaticIKOptions:
    tracking_weight: float = 1.0e8
    tension_weight: float = 1.0
    smoothness_weight: float = 5.0
    tension_min: float = 0.0
    tension_max: float = 120.0
    max_tension_step: float = 8.0
    optimizer_method: str = "Powell"
    optimizer_maxiter: int = 60
    optimizer_ftol: float = 1.0e-10

    # Inner PCS static solve options.
    # Inner PCS static solve options.
    # "linearized" uses one local linear PCS equilibrium around the previous
    # continuation state. This is the default because it is fast enough for
    # outer tension-optimization trajectories in VSCode.
    # "energy" minimizes total potential energy using residual(q) as the
    # gradient; it is more nonlinear but much slower.
    static_solver: str = "linearized"
    static_method: str = "L-BFGS-B"
    static_max_nfev: int = 120
    static_maxiter: int = 80
    static_residual_tol: float = 1.0e-5
    static_xtol: float = 1.0e-9
    static_ftol: float = 1.0e-9
    static_gtol: float = 1.0e-9
    fd_eps: float = 1.0e-6


@dataclass(frozen=True)
class PCSStaticSolveResult:
    q: Array
    success: bool
    message: str
    residual_norm: float
    cost: float
    nfev: int


@dataclass(frozen=True)
class PCSTensionStaticIKPointResult:
    target_yz: Array
    q: Array
    cable_lengths: Array
    cable_tensions: Array
    tip_yz: Array
    tracking_error: Array
    objective: float
    success: bool
    message: str
    solve_time_s: float
    static_result: PCSStaticSolveResult


@dataclass(frozen=True)
class PCSTensionStaticIKTrajectoryResult:
    target_yz: Array
    q: Array
    cable_lengths: Array
    cable_tensions: Array
    tip_yz: Array
    tracking_error: Array
    objective: Array
    success: Array
    messages: tuple[str, ...]
    solve_time_s: Array
    point_results: tuple[PCSTensionStaticIKPointResult, ...]


# -----------------------------------------------------------------------------
# Geometry and routing specs
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PCSCableRoutingSpec:
    """Station-based three-cable routing with station-dependent radii."""

    guide_stations: Array
    cable_angles: Array
    cable_radii: Array | float

    @classmethod
    def default(cls) -> "PCSCableRoutingSpec":
        return cls(
            guide_stations=1.0e-3 * np.array([0.0, 72.3, 152.1, 232.5, 312.1, 390.1], dtype=float),
            cable_angles=np.deg2rad(np.array([0.0, 120.0, 240.0], dtype=float)),
            cable_radii=1.0e-3 * np.array([30.0, 28.0, 26.0, 24.0, 22.0, 20.0], dtype=float),
        )


def _expand_station_radii(cable_radii: Array | float, *, n_stations: int) -> Array:
    radii = np.asarray(cable_radii, dtype=float)
    if radii.ndim == 0:
        expanded = np.full(n_stations, float(radii), dtype=float)
    elif radii.shape == (n_stations,):
        expanded = radii.copy()
    else:
        raise ValueError(f"cable_radii must be scalar or shape ({n_stations},); got {radii.shape}")
    if np.any(expanded < 0.0):
        raise ValueError("cable_radii must be nonnegative")
    return np.ascontiguousarray(expanded, dtype=float)


@dataclass(frozen=True)
class PCSBeamParameters:
    E: float
    G: float
    rho: float
    backbone_radius: float = 1.5e-3
    shear_correction: float = 5.0 / 6.0

    def section_stiffness(self) -> Array:
        A = np.pi * self.backbone_radius**2
        I = np.pi * self.backbone_radius**4 / 4.0
        J = np.pi * self.backbone_radius**4 / 2.0
        EA = self.E * A
        GA = self.shear_correction * self.G * A
        EI = self.E * I
        GJ = self.G * J
        # xi = [kx, ky, kz, vx, vy, vz], local x-axis is axial.
        return np.diag([GJ, EI, EI, EA, GA, GA]).astype(float)

    @property
    def mass_per_length(self) -> float:
        return float(self.rho * np.pi * self.backbone_radius**2)


# -----------------------------------------------------------------------------
# Lie algebra utilities for local x-axis Cosserat rod
# -----------------------------------------------------------------------------

def _skew(w: Array) -> Array:
    wx, wy, wz = w
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=float)


def _hat_se3(xi: Array) -> Array:
    H = np.zeros((4, 4), dtype=float)
    H[:3, :3] = _skew(xi[:3])
    H[:3, 3] = xi[3:6]
    return H


def _exp_xi(xi: Array, ds: float) -> Array:
    return expm(_hat_se3(xi) * float(ds))


@dataclass
class PCSSingleSegmentModel:
    """Single-segment PCS model used as the static IK comparison plant."""

    breakpoints: Array
    K_sections: Array
    q0: Array
    guide_stations: Array
    cable_offsets: Array  # shape (n_stations, n_cables, 3)
    cable_interval_mask: Optional[Array] = None  # shape (n_stations - 1, n_cables)
    disk_masses: Optional[Array] = None  # shape (n_stations,), centerline disk/lumped masses
    mass_per_length: float = 0.0
    gravity: Array = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        self.breakpoints = np.asarray(self.breakpoints, dtype=float)
        self.K_sections = np.asarray(self.K_sections, dtype=float)
        self.q0 = np.asarray(self.q0, dtype=float)
        self.guide_stations = np.asarray(self.guide_stations, dtype=float)
        self.cable_offsets = np.asarray(self.cable_offsets, dtype=float)
        if self.cable_interval_mask is not None:
            self.cable_interval_mask = np.asarray(self.cable_interval_mask, dtype=float)
        self.gravity = np.asarray(self.gravity, dtype=float)

        if self.breakpoints.ndim != 1 or self.breakpoints.size < 2:
            raise ValueError("breakpoints must be a one-dimensional array with at least two entries")
        if not np.all(np.diff(self.breakpoints) > 0.0):
            raise ValueError("breakpoints must be strictly increasing")
        self.N = self.breakpoints.size - 1
        self.segment_lengths = np.diff(self.breakpoints)
        self.length = float(self.breakpoints[-1] - self.breakpoints[0])
        self.dof = 6 * self.N

        if self.K_sections.shape == (6, 6):
            self.K_sections = np.repeat(self.K_sections[None, :, :], self.N, axis=0)
        if self.K_sections.shape != (self.N, 6, 6):
            raise ValueError(f"K_sections must have shape (6,6) or ({self.N},6,6); got {self.K_sections.shape}")
        if self.q0.shape != (self.dof,):
            raise ValueError(f"q0 must have shape ({self.dof},); got {self.q0.shape}")
        if self.cable_offsets.ndim != 3 or self.cable_offsets.shape[0] != self.guide_stations.size or self.cable_offsets.shape[2] != 3:
            raise ValueError("cable_offsets must have shape (n_stations, n_cables, 3)")
        self.n_cables = self.cable_offsets.shape[1]
        if self.cable_interval_mask is not None:
            expected_shape = (self.guide_stations.size - 1, self.n_cables)
            if self.cable_interval_mask.shape != expected_shape:
                raise ValueError(
                    f"cable_interval_mask must have shape {expected_shape}; got {self.cable_interval_mask.shape}"
                )

        if self.disk_masses is not None:
            self.disk_masses = np.asarray(self.disk_masses, dtype=float)
            if self.disk_masses.shape != (self.guide_stations.size,):
                raise ValueError(
                    "disk_masses must have shape "
                    f"({self.guide_stations.size},); got {self.disk_masses.shape}"
                )
            if np.any(self.disk_masses < 0.0):
                raise ValueError("disk_masses must be nonnegative")

        if self.gravity.shape != (3,):
            raise ValueError(f"gravity must have shape (3,), got {self.gravity.shape}")

        self.K_pcs = self._build_K_pcs()

    @property
    def base_yz(self) -> Array:
        return self.tip_position(self.q0)[1:3].copy()

    def initial_state(self) -> Array:
        return self.q0.copy()

    def _build_K_pcs(self) -> Array:
        K = np.zeros((self.dof, self.dof), dtype=float)
        for i in range(self.N):
            block = self.segment_lengths[i] * self.K_sections[i]
            K[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)] = block
        return K

    def _xis(self, q: Array) -> Array:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"q must have shape ({self.dof},); got {q.shape}")
        return q.reshape(self.N, 6)

    def pose_at(self, q: Array, s: float) -> Array:
        """Pose at arc-length station s in world frame."""
        s = float(s)
        if s < self.breakpoints[0] - 1e-12 or s > self.breakpoints[-1] + 1e-12:
            raise ValueError(f"station {s} lies outside [{self.breakpoints[0]}, {self.breakpoints[-1]}]")
        xis = self._xis(q)
        g = np.eye(4)
        remaining = s - self.breakpoints[0]
        if remaining <= 0.0:
            return g
        for i, ds_i in enumerate(self.segment_lengths):
            ds_use = min(float(ds_i), remaining)
            if ds_use > 0.0:
                g = g @ _exp_xi(xis[i], ds_use)
            remaining -= ds_use
            if remaining <= 1e-14:
                break
        return g

    def node_poses(self, q: Array) -> Array:
        poses = np.zeros((self.N + 1, 4, 4), dtype=float)
        poses[0] = np.eye(4)
        g = np.eye(4)
        for i, xi in enumerate(self._xis(q)):
            g = g @ _exp_xi(xi, self.segment_lengths[i])
            poses[i + 1] = g
        return poses

    def tip_position(self, q: Array) -> Array:
        return self.pose_at(q, self.breakpoints[-1])[:3, 3]

    def centerline_midpoints(self, q: Array) -> Array:
        poses = self.node_poses(q)
        nodes = poses[:, :3, 3]
        return 0.5 * (nodes[:-1] + nodes[1:])

    def cable_points(self, q: Array) -> Array:
        """Return guide-point positions, shape (n_stations, n_cables, 3)."""
        pts = np.zeros((self.guide_stations.size, self.n_cables, 3), dtype=float)
        for a, s in enumerate(self.guide_stations):
            g = self.pose_at(q, float(s))
            R = g[:3, :3]
            p = g[:3, 3]
            pts[a] = p[None, :] + self.cable_offsets[a] @ R.T
        return pts

    def guide_station_positions(self, q: Array) -> Array:
        """Centerline positions of disk/guide stations, shape (n_stations, 3)."""
        pts = np.zeros((self.guide_stations.size, 3), dtype=float)
        for a, s in enumerate(self.guide_stations):
            pts[a] = self.pose_at(q, float(s))[:3, 3]
        return pts

    def cable_lengths(self, q: Array) -> Array:
        pts = self.cable_points(q)
        diffs = pts[1:] - pts[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=2)
        if self.cable_interval_mask is not None:
            segment_lengths = segment_lengths * self.cable_interval_mask
        return segment_lengths.sum(axis=0)

    def cable_length_jacobian_fd(self, q: Array, eps: float = 1.0e-6) -> Array:
        q = np.asarray(q, dtype=float)
        A = np.zeros((self.n_cables, self.dof), dtype=float)
        for k in range(self.dof):
            dq = np.zeros_like(q)
            dq[k] = eps
            lp = self.cable_lengths(q + dq)
            lm = self.cable_lengths(q - dq)
            A[:, k] = (lp - lm) / (2.0 * eps)
        return A

    def gravity_generalized_force_fd(self, q: Array, eps: float = 1.0e-6) -> Array:
        no_disk_masses = self.disk_masses is None or np.all(self.disk_masses == 0.0)
        if np.linalg.norm(self.gravity) == 0.0 or (no_disk_masses and self.mass_per_length <= 0.0):
            return np.zeros(self.dof, dtype=float)

        gravity = self.gravity.copy()

        def potential(q_eval: Array) -> float:
            V = 0.0
            if self.mass_per_length > 0.0 and np.linalg.norm(gravity) > 0.0:
                mids = self.centerline_midpoints(q_eval)
                masses = self.mass_per_length * self.segment_lengths
                V += -float(np.sum(masses[:, None] * mids * gravity[None, :]))
            if self.disk_masses is not None and np.linalg.norm(gravity) > 0.0:
                disk_pts = self.guide_station_positions(q_eval)
                V += -float(np.sum(self.disk_masses[:, None] * disk_pts * gravity[None, :]))
            return V

        grad = np.zeros(self.dof, dtype=float)
        q = np.asarray(q, dtype=float)
        for k in range(self.dof):
            dq = np.zeros_like(q)
            dq[k] = eps
            grad[k] = (potential(q + dq) - potential(q - dq)) / (2.0 * eps)
        return -grad

    def gravity_potential(self, q: Array) -> float:
        """Gravitational potential used by the PCS energy solver."""
        if np.linalg.norm(self.gravity) == 0.0:
            return 0.0

        V = 0.0
        if self.mass_per_length > 0.0:
            mids = self.centerline_midpoints(q)
            masses = self.mass_per_length * self.segment_lengths
            V += -float(np.sum(masses[:, None] * mids * self.gravity[None, :]))
        if self.disk_masses is not None and not np.all(self.disk_masses == 0.0):
            disk_pts = self.guide_station_positions(q)
            V += -float(np.sum(self.disk_masses[:, None] * disk_pts * self.gravity[None, :]))
        return V

    def elastic_energy(self, q: Array) -> float:
        dq = np.asarray(q, dtype=float) - self.q0
        return 0.5 * float(dq @ (self.K_pcs @ dq))

    def total_potential(self, q: Array, cable_tensions: Array) -> float:
        """Total potential Pi(q) whose gradient is static_residual(q).

        Pi = elastic strain energy + tendon tension potential + gravity potential
           = 0.5 dq^T K dq + tau^T l(q) + V_g(q)

        With the convention delta W_tendon = -tau^T delta l, the equilibrium
        condition grad(Pi)=0 is exactly
            K(q-q0) + A(q)^T tau - Q_g(q) = 0.
        """
        tau = np.asarray(cable_tensions, dtype=float)
        return self.elastic_energy(q) + float(tau @ self.cable_lengths(q)) + self.gravity_potential(q)

    def static_residual(self, q: Array, cable_tensions: Array, fd_eps: float = 1.0e-6) -> Array:
        tau = np.asarray(cable_tensions, dtype=float)
        if tau.shape != (self.n_cables,):
            raise ValueError(f"cable_tensions must have shape ({self.n_cables},); got {tau.shape}")
        elastic = self.K_pcs @ (np.asarray(q, dtype=float) - self.q0)
        A = self.cable_length_jacobian_fd(q, eps=fd_eps)
        gravity = self.gravity_generalized_force_fd(q, eps=fd_eps)
        return elastic + A.T @ tau - gravity

    def solve_static(self, cable_tensions: Array, q0: Optional[Array] = None, options: Optional[PCSTensionStaticIKOptions] = None) -> PCSStaticSolveResult:
        opts = PCSTensionStaticIKOptions() if options is None else options
        q_start = self.initial_state() if q0 is None else np.asarray(q0, dtype=float).copy()

        if str(opts.static_solver).lower() == "energy":
            res = minimize(
                lambda x: self.total_potential(x, cable_tensions),
                q_start,
                jac=lambda x: self.static_residual(x, cable_tensions, fd_eps=opts.fd_eps),
                method=opts.static_method,
                options={
                    "maxiter": int(opts.static_maxiter),
                    "maxfun": int(opts.static_max_nfev),
                    "ftol": float(opts.static_ftol),
                    "gtol": float(opts.static_gtol),
                },
            )
            r = self.static_residual(res.x, cable_tensions, fd_eps=opts.fd_eps)
            residual_norm = float(np.linalg.norm(r))
            success = bool(res.success or residual_norm <= float(opts.static_residual_tol))
            return PCSStaticSolveResult(
                q=res.x.copy(),
                success=success,
                message=str(res.message),
                residual_norm=residual_norm,
                cost=float(res.fun),
                nfev=int(getattr(res, "nfev", 0)),
            )

        # Fallback/debug mode. This is mathematically valid but much slower for
        # this implementation because least_squares estimates another numerical
        # Jacobian around a residual that already computes finite differences.
        res = least_squares(
            lambda x: self.static_residual(x, cable_tensions, fd_eps=opts.fd_eps),
            q_start,
            max_nfev=opts.static_max_nfev,
            xtol=opts.static_xtol,
            ftol=opts.static_ftol,
            gtol=opts.static_gtol,
        )
        r = self.static_residual(res.x, cable_tensions, fd_eps=opts.fd_eps)
        return PCSStaticSolveResult(
            q=res.x.copy(),
            success=bool(res.success),
            message=str(res.message),
            residual_norm=float(np.linalg.norm(r)),
            cost=float(res.cost),
            nfev=int(res.nfev),
        )


class PCSTwoSegmentsModel(PCSSingleSegmentModel):
    """Two-section PCS model with three independently routed cables per section.

    This class intentionally reuses the PCS kinematics and static-equilibrium
    operations from ``PCSSingleSegmentModel``. The two-section behavior is
    encoded by a cable interval mask:

    - cables 0..2 contribute only over section 1 by default,
    - cables 3..5 contribute only over section 2 by default.

    The undeformed backbone axis remains the model x-axis, matching the
    inherited Cosserat/PCS convention.
    """

    @classmethod
    def with_uniform_section_cables(
        cls,
        *,
        section_lengths: Array,
        section_radii: Array,
        segments_per_section: int,
        section_stiffness: Array,
        cable_angles: Array | None = None,
        swap_section_cable_order: bool = False,
        disk_masses: float | Array | None = None,
        mass_per_length: float = 0.0,
        gravity: Array | None = None,
    ) -> "PCSTwoSegmentsModel":
        lengths = np.asarray(section_lengths, dtype=float).reshape(2)
        radii = np.asarray(section_radii, dtype=float).reshape(2)
        if np.any(lengths <= 0.0):
            raise ValueError("section_lengths must be positive")
        if np.any(radii < 0.0):
            raise ValueError("section_radii must be nonnegative")

        segments_per_section = int(segments_per_section)
        if segments_per_section <= 0:
            raise ValueError("segments_per_section must be positive")

        angles = (
            np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=float)
            if cable_angles is None
            else np.asarray(cable_angles, dtype=float).reshape(3)
        )

        section_1 = np.linspace(0.0, lengths[0], segments_per_section + 1)
        section_2 = np.linspace(lengths[0], float(np.sum(lengths)), segments_per_section + 1)[1:]
        breakpoints = np.concatenate([section_1, section_2])
        guide_stations = breakpoints.copy()

        n_cables = 6
        section_1_cable_slice = slice(3, 6) if swap_section_cable_order else slice(0, 3)
        section_2_cable_slice = slice(0, 3) if swap_section_cable_order else slice(3, 6)
        cable_offsets = np.zeros((guide_stations.size, n_cables, 3), dtype=float)
        for cable_idx, angle in enumerate(angles):
            proximal_idx = section_1_cable_slice.start + cable_idx
            section_1_mask = guide_stations <= lengths[0] + 1.0e-12
            cable_offsets[section_1_mask, proximal_idx, 1] = radii[0] * np.cos(angle)
            cable_offsets[section_1_mask, proximal_idx, 2] = radii[0] * np.sin(angle)

            distal_idx = section_2_cable_slice.start + cable_idx
            section_2_mask = guide_stations >= lengths[0] - 1.0e-12
            cable_offsets[section_2_mask, distal_idx, 1] = radii[1] * np.cos(angle)
            cable_offsets[section_2_mask, distal_idx, 2] = radii[1] * np.sin(angle)

        interval_midpoints = 0.5 * (guide_stations[:-1] + guide_stations[1:])
        cable_interval_mask = np.zeros((guide_stations.size - 1, n_cables), dtype=float)
        cable_interval_mask[interval_midpoints <= lengths[0] + 1.0e-12, section_1_cable_slice] = 1.0
        cable_interval_mask[interval_midpoints >= lengths[0] - 1.0e-12, section_2_cable_slice] = 1.0

        total_segments = 2 * segments_per_section
        xi0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
        q0 = np.tile(xi0, total_segments)
        masses = _expand_station_masses(disk_masses, n_stations=guide_stations.size)
        gravity_vec = np.zeros(3, dtype=float) if gravity is None else np.asarray(gravity, dtype=float)
        if gravity_vec.shape != (3,):
            raise ValueError(f"gravity must have shape (3,), got {gravity_vec.shape}")

        model = cls(
            breakpoints=breakpoints,
            K_sections=section_stiffness,
            q0=q0,
            guide_stations=guide_stations,
            cable_offsets=cable_offsets,
            cable_interval_mask=cable_interval_mask,
            disk_masses=masses,
            mass_per_length=float(mass_per_length),
            gravity=gravity_vec,
        )
        model.section_lengths = lengths.copy()
        model.section_radii = radii.copy()
        model.segments_per_section = segments_per_section
        model.section_boundary_station = float(lengths[0])
        model.swap_section_cable_order = bool(swap_section_cable_order)
        return model

    @property
    def section_end_indices(self) -> tuple[int, int]:
        return (int(self.segments_per_section), int(2 * self.segments_per_section))

    def section_end_positions(self, q: Array) -> Array:
        poses = self.node_poses(q)
        first_idx, second_idx = self.section_end_indices
        return np.vstack([poses[first_idx, :3, 3], poses[second_idx, :3, 3]])


def _subdivide_guide_station_intervals(guide_stations: Array, *, n_segments: int) -> Array:
    """Create PCS breakpoints by subdividing each guide-station interval.

    For the default six identified guide stations and n_segments=10, this
    produces two PCS constant-strain elements per guide-station interval:
        [s0, mid01, s1, mid12, s2, ..., s5]

    If n_segments is not an integer multiple of the number of guide intervals,
    fall back to uniform subdivision over the total length.
    """
    guide_stations = np.asarray(guide_stations, dtype=float)
    n_intervals = guide_stations.size - 1
    if n_intervals <= 0:
        raise ValueError("guide_stations must contain at least two stations")

    n_segments = int(n_segments)
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    if n_segments % n_intervals != 0:
        return np.linspace(float(guide_stations[0]), float(guide_stations[-1]), n_segments + 1)

    pieces_per_interval = n_segments // n_intervals
    chunks = []
    for i in range(n_intervals):
        local = np.linspace(guide_stations[i], guide_stations[i + 1], pieces_per_interval + 1)
        if i > 0:
            local = local[1:]
        chunks.append(local)
    return np.concatenate(chunks).astype(float)


def _expand_station_masses(masses: float | Array | None, *, n_stations: int) -> Array:
    """Expand scalar or station-wise masses to shape (n_stations,)."""
    if masses is None:
        return np.zeros(n_stations, dtype=float)
    arr = np.asarray(masses, dtype=float)
    if arr.ndim == 0:
        expanded = np.full(n_stations, float(arr), dtype=float)
    elif arr.shape == (n_stations,):
        expanded = arr.copy()
    else:
        raise ValueError(f"masses must be scalar or have shape ({n_stations},); got {arr.shape}")
    if np.any(expanded < 0.0):
        raise ValueError("masses must be nonnegative")
    return np.ascontiguousarray(expanded, dtype=float)


def build_default_single_segment_pcs_ik_model(
    root: Path,
    *,
    n_segments: int = 10,
    cable_routing: Optional[PCSCableRoutingSpec] = None,
    enable_gravity: bool = False,
    gravity_vector: Optional[Array] = None,
    disk_masses: float | Array | None = None,
) -> PCSSingleSegmentModel:
    """Build a PCS comparison model matching the existing single-segment setup.

    It uses the same default guide stations, cable angles, station radii, backbone
    radius and identified E/rho convention as ``single_segment.py``. If the
    identification H5 file is unavailable, pass parameters manually by editing
    this function or replace the fallback values below.
    """

    # Safe fallback for code execution outside your project. Replace with the
    # identified values when running formal comparisons.
    E = 0.875e6
    rho = 1050

    nu = 0.47
    G = E / (2.0 * (1.0 + nu))
    params = PCSBeamParameters(E=E, G=G, rho=rho, backbone_radius=1.5e-3)
    K = params.section_stiffness()

    spec = PCSCableRoutingSpec.default() if cable_routing is None else cable_routing
    guide_stations = np.asarray(spec.guide_stations, dtype=float)
    angles = np.asarray(spec.cable_angles, dtype=float)
    station_radii = _expand_station_radii(spec.cable_radii, n_stations=guide_stations.size)

    cable_offsets = np.zeros((guide_stations.size, angles.size, 3), dtype=float)
    for j, angle in enumerate(angles):
        cable_offsets[:, j, 1] = station_radii * np.cos(angle)
        cable_offsets[:, j, 2] = station_radii * np.sin(angle)

    breakpoints = _subdivide_guide_station_intervals(guide_stations, n_segments=int(n_segments))
    xi0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    q0 = np.tile(xi0, int(n_segments))

    disk_mass_array = _expand_station_masses(disk_masses, n_stations=guide_stations.size)

    if gravity_vector is None:
        gravity_vec = np.array([9.81, 0.0, 0.0], dtype=float)
    else:
        gravity_vec = np.asarray(gravity_vector, dtype=float)
        if gravity_vec.shape != (3,):
            raise ValueError(f"gravity_vector must have shape (3,), got {gravity_vec.shape}")
    gravity = gravity_vec if enable_gravity else np.zeros(3, dtype=float)

    return PCSSingleSegmentModel(
        breakpoints=breakpoints,
        K_sections=K,
        q0=q0,
        guide_stations=guide_stations,
        cable_offsets=cable_offsets,
        disk_masses=disk_mass_array,
        mass_per_length=params.mass_per_length if enable_gravity else 0.0,
        gravity=gravity,
    )


# -----------------------------------------------------------------------------
# PCS tension-static IK, parallel to your existing optimizer
# -----------------------------------------------------------------------------

def solve_tip_yz_pcs_tension_static_inverse(
    *,
    model: PCSSingleSegmentModel,
    target_yz: Array,
    options: Optional[PCSTensionStaticIKOptions] = None,
    initial_tensions: Optional[Array] = None,
    previous_tensions: Optional[Array] = None,
    q0: Optional[Array] = None,
) -> PCSTensionStaticIKPointResult:
    point_t0 = time.perf_counter()
    opts = PCSTensionStaticIKOptions() if options is None else options
    target = np.asarray(target_yz, dtype=float)
    if target.shape != (2,):
        raise ValueError(f"target_yz must have shape (2,), got {target.shape}")

    q_start = model.initial_state() if q0 is None else np.asarray(q0, dtype=float).copy()
    if q_start.shape != (model.dof,):
        raise ValueError(f"q0 must have shape ({model.dof},), got {q_start.shape}")

    if initial_tensions is None:
        tau0 = np.full(model.n_cables, max(float(opts.tension_min), 0.0), dtype=float)
    else:
        tau0 = np.asarray(initial_tensions, dtype=float).copy()
    if tau0.shape != (model.n_cables,):
        raise ValueError(f"initial_tensions must have shape ({model.n_cables},), got {tau0.shape}")

    tau_prev = tau0 if previous_tensions is None else np.asarray(previous_tensions, dtype=float).copy()
    if tau_prev.shape != (model.n_cables,):
        raise ValueError(f"previous_tensions must have shape ({model.n_cables},), got {tau_prev.shape}")

    lower = np.maximum(float(opts.tension_min), tau_prev - float(opts.max_tension_step))
    upper = np.minimum(float(opts.tension_max), tau_prev + float(opts.max_tension_step))
    tau0 = np.clip(tau0, lower, upper)
    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]

    cache: dict[tuple[float, ...], tuple[float, PCSStaticSolveResult, Array]] = {}

    linearized_mode = str(opts.static_solver).lower() == "linearized"
    if linearized_mode:
        # Precompute the local tendon-length Jacobian once for this continuation
        # target. During the outer tension optimization q(tau) is then obtained
        # from the local equilibrium
        #     K(q-q_nat) + A_start^T tau - Qg_start = 0.
        # This avoids recomputing a 60-variable finite-difference Jacobian for
        # every Powell objective evaluation.
        A_start = model.cable_length_jacobian_fd(q_start, eps=opts.fd_eps)
        Qg_start = model.gravity_generalized_force_fd(q_start, eps=opts.fd_eps)
        K_diag = np.diag(model.K_pcs).copy()
        if np.any(np.abs(K_diag) < 1.0e-18):
            raise ValueError("PCS stiffness diagonal contains near-zero entries")

    def evaluate(tau_trial: Array) -> tuple[float, PCSStaticSolveResult, Array]:
        tau_i = np.asarray(tau_trial, dtype=float)
        key = tuple(np.round(tau_i, decimals=10))
        if key in cache:
            return cache[key]

        if linearized_mode:
            q_static = model.q0 - (A_start.T @ tau_i - Qg_start) / K_diag
            static = PCSStaticSolveResult(
                q=q_static.copy(),
                success=True,
                message="linearized PCS static equilibrium",
                residual_norm=float("nan"),
                cost=float("nan"),
                nfev=1,
            )
        else:
            static = model.solve_static(tau_i, q0=q_start, options=opts)
        yz_i = model.tip_position(static.q)[1:3]
        tracking = yz_i - target
        dtau = tau_i - tau_prev
        objective = opts.tracking_weight * float(tracking @ tracking)
        objective += opts.tension_weight * float(tau_i @ tau_i)
        objective += opts.smoothness_weight * float(dtau @ dtau)
        if not static.success:
            objective += 1.0e6 * static.residual_norm
        value = (float(objective), static, yz_i)
        cache[key] = value
        return value

    def objective(tau_trial: Array) -> float:
        try:
            return evaluate(tau_trial)[0]
        except Exception:
            return 1.0e100

    solver_options = {"maxiter": int(opts.optimizer_maxiter)}
    if opts.optimizer_method.upper() in {"SLSQP", "L-BFGS-B", "TNC"}:
        solver_options["ftol"] = float(opts.optimizer_ftol)

    opt = minimize(objective, tau0, method=opts.optimizer_method, bounds=bounds, options=solver_options)
    objective_value, static_result, yz = evaluate(opt.x)
    if linearized_mode:
        final_residual = model.static_residual(static_result.q, np.asarray(opt.x, dtype=float), fd_eps=opts.fd_eps)
        static_result = PCSStaticSolveResult(
            q=static_result.q.copy(),
            success=True,
            message=static_result.message,
            residual_norm=float(np.linalg.norm(final_residual)),
            cost=float("nan"),
            nfev=1,
        )
    lengths = model.cable_lengths(static_result.q)

    return PCSTensionStaticIKPointResult(
        target_yz=target.copy(),
        q=static_result.q.copy(),
        cable_lengths=lengths.copy(),
        cable_tensions=np.asarray(opt.x, dtype=float).copy(),
        tip_yz=yz.copy(),
        tracking_error=yz - target,
        objective=objective_value,
        success=bool(opt.success and static_result.success),
        message=f"outer: {opt.message}; inner: {static_result.message}",
        solve_time_s=float(time.perf_counter() - point_t0),
        static_result=static_result,
    )


def solve_tip_yz_pcs_tension_static_trajectory(
    *,
    model: PCSSingleSegmentModel,
    targets_yz: Array,
    options: Optional[PCSTensionStaticIKOptions] = None,
    initial_tensions: Optional[Array] = None,
    q0: Optional[Array] = None,
) -> PCSTensionStaticIKTrajectoryResult:
    targets = np.asarray(targets_yz, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(f"targets_yz must have shape (n_points,2), got {targets.shape}")

    point_results: list[PCSTensionStaticIKPointResult] = []
    q_start = model.initial_state() if q0 is None else np.asarray(q0, dtype=float).copy()
    tau_start = np.zeros(model.n_cables, dtype=float) if initial_tensions is None else np.asarray(initial_tensions, dtype=float).copy()

    for i, target in enumerate(targets):
        result = solve_tip_yz_pcs_tension_static_inverse(
            model=model,
            target_yz=target,
            options=options,
            initial_tensions=tau_start,
            previous_tensions=tau_start,
            q0=q_start,
        )
        point_results.append(result)
        q_start = result.q
        tau_start = result.cable_tensions
        print(
            f"PCS point {i + 1}/{targets.shape[0]}: "
            f"success={result.success}, "
            f"err_mm={1.0e3 * np.linalg.norm(result.tracking_error):.3f}, "
            f"time_s={result.solve_time_s:.2f}, "
            f"tau={np.array2string(result.cable_tensions, precision=3)}"
        )

    return PCSTensionStaticIKTrajectoryResult(
        target_yz=targets.copy(),
        q=np.column_stack([r.q for r in point_results]),
        cable_lengths=np.column_stack([r.cable_lengths for r in point_results]),
        cable_tensions=np.column_stack([r.cable_tensions for r in point_results]),
        tip_yz=np.vstack([r.tip_yz for r in point_results]),
        tracking_error=np.vstack([r.tracking_error for r in point_results]),
        objective=np.asarray([r.objective for r in point_results], dtype=float),
        success=np.asarray([r.success for r in point_results], dtype=bool),
        messages=tuple(r.message for r in point_results),
        solve_time_s=np.asarray([r.solve_time_s for r in point_results], dtype=float),
        point_results=tuple(point_results),
    )
