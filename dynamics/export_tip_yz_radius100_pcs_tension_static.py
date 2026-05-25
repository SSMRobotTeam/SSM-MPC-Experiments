from __future__ import annotations

"""Export PCS tension-static IK curves for selectable tip y-z paths.

VSCode direct-run version: edit the configuration constants below, then click
Run. No command-line arguments are required. Run the GEBF script first with the
same trajectory settings if you want automatic comparisons.
"""

from pathlib import Path
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inverse_kinematics.pcs_tension_static_optimizer import (  # noqa: E402
    PCSTensionStaticIKOptions,
    build_default_single_segment_pcs_ik_model,
    solve_tip_yz_pcs_tension_static_trajectory,
)
from inverse_kinematics.trajectory_targets import make_tip_yz_trajectory  # noqa: E402


# -----------------------------------------------------------------------------
# VSCode direct-run configuration
# -----------------------------------------------------------------------------
# Supported trajectories: "circle", "figure8", "spiral".
TRAJECTORY = "circle"
RADIUS = 100.0e-3
N_CIRCLE = 72
N_LINE = 20
N_FIGURE8 = 144
N_SPIRAL = 144
SPIRAL_TURNS = 2.0

# Gravity is disabled by default for model-comparison runs. When enabled, both
# GEBF and PCS should use the same vector. Current convention: global +x.
ENABLE_GRAVITY = True
GRAVITY_VECTOR = np.array([0, 9.81, 0.0], dtype=float)

# PCS discretization: with identified guide stations, n_segments=10 means two
# PCS subsegments between every adjacent pair of cable-guide stations.
PCS_SEGMENTS = 5

# Centerline disk/lumped masses at the identified guide stations, kg.
# The six entries correspond to s = [0, 72.3, 152.1, 232.5, 312.1, 390.1] mm.
# Keep these identical to export_tip_yz_radius100_tension_static.py.
DISK_MASSES = np.array([0.0, 0.017, 0.015, 0.017, 0.014, 0.027+0.05], dtype=float)



def main() -> None:
    gravity_vector = np.asarray(GRAVITY_VECTOR, dtype=float)
    disk_masses = np.asarray(DISK_MASSES, dtype=float)
    model = build_default_single_segment_pcs_ik_model(
        ROOT,
        n_segments=PCS_SEGMENTS,
        enable_gravity=bool(ENABLE_GRAVITY),
        disk_masses=disk_masses,
        gravity_vector=gravity_vector,
    )

    q0 = model.q0
    base_yz = model.base_yz
    traj = make_tip_yz_trajectory(
        base_yz,
        trajectory=TRAJECTORY,
        radius=RADIUS,
        n_circle=N_CIRCLE,
        n_line=N_LINE,
        n_figure8=N_FIGURE8,
        n_spiral=N_SPIRAL,
        spiral_turns=SPIRAL_TURNS,
    )
    targets = traj.targets
    segment = traj.segment
    progress = traj.progress

    out_dir = ROOT / "outputs" / "inverse_kinematics" / f"tip_yz_{traj.name}_pcs_tension_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    options = PCSTensionStaticIKOptions(
        tracking_weight=1.0e8,
        tension_weight=0.1,
        smoothness_weight=1.0,
        tension_min=0.0,
        tension_max=80.0,
        max_tension_step=8.0,
        optimizer_maxiter=80,
        static_solver="linearized",
        static_max_nfev=120,
        static_maxiter=80,
        fd_eps=1.0e-6,
    )

    wall_t0 = time.perf_counter()
    result = solve_tip_yz_pcs_tension_static_trajectory(
        model=model,
        targets_yz=targets,
        options=options,
        initial_tensions=np.zeros(model.n_cables, dtype=float),
        q0=q0,
    )
    wall_time_total_s = float(time.perf_counter() - wall_t0)

    history_path = out_dir / f"tip_yz_{traj.name}_pcs_tension_static_ik_history.h5"
    with h5py.File(history_path, "w") as h5:
        h5.attrs["radius"] = float(RADIUS)
        h5.attrs["trajectory"] = str(TRAJECTORY)
        h5.attrs["path_definition"] = traj.definition
        h5.attrs["solver"] = "continuation PCS tension-static IK"
        h5.attrs["pcs_n_segments"] = int(model.N)
        h5.attrs["gravity_enabled"] = bool(ENABLE_GRAVITY)
        h5.attrs["gravity_vector"] = gravity_vector
        h5.attrs["disk_masses"] = disk_masses
        h5.attrs["wall_time_total_s"] = wall_time_total_s
        h5.attrs["solve_time_total_s"] = float(np.sum(result.solve_time_s))
        h5.attrs["solve_time_mean_s"] = float(np.mean(result.solve_time_s))
        h5.create_dataset("guide_stations", data=model.guide_stations)
        h5.create_dataset("pcs_breakpoints", data=model.breakpoints)
        h5.create_dataset("disk_masses", data=disk_masses)
        h5.create_dataset("target_yz", data=result.target_yz)
        h5.create_dataset("tip_yz", data=result.tip_yz)
        h5.create_dataset("tracking_error_yz", data=result.tracking_error)
        h5.create_dataset("path_progress", data=progress)
        h5.create_dataset("segment", data=segment)
        h5.create_dataset("q", data=result.q)
        h5.create_dataset("cable_lengths", data=result.cable_lengths)
        h5.create_dataset("cable_tensions", data=result.cable_tensions)
        h5.create_dataset("solve_time_s", data=result.solve_time_s)
        h5.create_dataset("optimizer_success", data=result.success)
        h5.create_dataset("objective", data=result.objective)

    csv_path = out_dir / f"tip_yz_{traj.name}_pcs_tension_static_cable_lengths.csv"
    header = (
        "progress_mm,segment,target_y_mm,target_z_mm,tip_y_mm,tip_z_mm,"
        "length1_mm,length2_mm,length3_mm,tension1,tension2,tension3,solve_time_s,success"
    )
    csv_data = np.column_stack(
        [
            1.0e3 * progress,
            segment,
            1.0e3 * result.target_yz,
            1.0e3 * result.tip_yz,
            1.0e3 * result.cable_lengths.T,
            result.cable_tensions.T,
            result.solve_time_s,
            result.success.astype(float),
        ]
    )
    np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="")

    fig_len, ax_len = plt.subplots(figsize=(7.8, 4.8))
    for j in range(result.cable_lengths.shape[0]):
        ax_len.plot(1.0e3 * progress, 1.0e3 * result.cable_lengths[j], label=f"cable {j + 1}")
    ax_len.set_xlabel("Path progress (mm)")
    ax_len.set_ylabel("Cable length (mm)")
    ax_len.grid(True)
    ax_len.legend()
    length_plot_path = out_dir / f"tip_yz_{traj.name}_pcs_tension_static_cable_lengths.png"
    fig_len.savefig(length_plot_path, dpi=220, bbox_inches="tight")

    fig_tau, ax_tau = plt.subplots(figsize=(7.8, 4.8))
    for j in range(result.cable_tensions.shape[0]):
        ax_tau.plot(1.0e3 * progress, result.cable_tensions[j], label=f"cable {j + 1}")
    ax_tau.set_xlabel("Path progress (mm)")
    ax_tau.set_ylabel("Cable tension / multiplier")
    ax_tau.grid(True)
    ax_tau.legend()
    tension_plot_path = out_dir / f"tip_yz_{traj.name}_pcs_tension_static_tensions.png"
    fig_tau.savefig(tension_plot_path, dpi=220, bbox_inches="tight")

    fig_path, ax_path = plt.subplots(figsize=(6.2, 6.2))
    ax_path.plot(
        1.0e3 * (result.target_yz[:, 0] - base_yz[0]),
        1.0e3 * (result.target_yz[:, 1] - base_yz[1]),
        "--",
        label="target",
    )
    ax_path.plot(
        1.0e3 * (result.tip_yz[:, 0] - base_yz[0]),
        1.0e3 * (result.tip_yz[:, 1] - base_yz[1]),
        label="PCS static IK",
    )
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("Relative tip y (mm)")
    ax_path.set_ylabel("Relative tip z (mm)")
    ax_path.grid(True)
    ax_path.legend()
    path_plot_path = out_dir / f"tip_yz_{traj.name}_pcs_tension_static_path.png"
    fig_path.savefig(path_plot_path, dpi=220, bbox_inches="tight")

    fig_time, ax_time = plt.subplots(figsize=(7.8, 4.8))
    ax_time.plot(1.0e3 * progress, result.solve_time_s, label="PCS")
    ax_time.set_xlabel("Path progress (mm)")
    ax_time.set_ylabel("Solve time per target (s)")
    ax_time.grid(True)
    ax_time.legend()
    time_plot_path = out_dir / f"tip_yz_{traj.name}_pcs_tension_static_solve_time.png"
    fig_time.savefig(time_plot_path, dpi=220, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Optional comparison against the original GEBF tension-static result.
    # Run export_tip_yz_radius100_tension_static.py first with the same trajectory.
    # ------------------------------------------------------------------
    gebf_history_path = (
        ROOT
        / "outputs"
        / "inverse_kinematics"
        / f"tip_yz_{traj.name}_tension_static"
        / f"tip_yz_{traj.name}_tension_static_ik_history.h5"
    )
    cable_length_compare_path = None
    cable_length_error_path = None
    tension_compare_path = None
    solve_time_compare_path = None
    solve_time_total_compare_path = None

    if gebf_history_path.exists():
        with h5py.File(gebf_history_path, "r") as h5:
            gebf_cable_lengths = h5["cable_lengths"][()]
            gebf_cable_tensions = h5["cable_tensions"][()]
            gebf_solve_time_s = h5["solve_time_s"][()] if "solve_time_s" in h5 else None
            gebf_solve_time_total_s = float(h5.attrs.get("solve_time_total_s", np.nan))
            gebf_gravity_enabled = bool(h5.attrs.get("gravity_enabled", False))
            gebf_gravity_vector = np.asarray(h5.attrs.get("gravity_vector", np.array([np.nan, np.nan, np.nan])), dtype=float)
            if "disk_masses" in h5:
                gebf_disk_masses = h5["disk_masses"][()]
            else:
                gebf_disk_masses = np.asarray(h5.attrs.get("disk_masses", np.full_like(disk_masses, np.nan)), dtype=float)

        if gebf_gravity_enabled != bool(ENABLE_GRAVITY) or not np.allclose(gebf_gravity_vector, gravity_vector, equal_nan=False):
            print(
                "Warning: GEBF and PCS gravity settings differ. "
                f"GEBF enabled={gebf_gravity_enabled}, vector={gebf_gravity_vector}; "
                f"PCS enabled={bool(ENABLE_GRAVITY)}, vector={gravity_vector}."
            )

        if not np.allclose(gebf_disk_masses, disk_masses, equal_nan=False):
            print(
                "Warning: GEBF and PCS disk masses differ. "
                f"GEBF={gebf_disk_masses}; PCS={disk_masses}."
            )

        with h5py.File(history_path, "a") as h5:
            h5.attrs["gebf_history_path"] = str(gebf_history_path)

        if gebf_cable_lengths.shape == result.cable_lengths.shape:
            cable_length_error = result.cable_lengths - gebf_cable_lengths

            with h5py.File(history_path, "a") as h5:
                h5.create_dataset("gebf_cable_lengths", data=gebf_cable_lengths)
                h5.create_dataset("pcs_minus_gebf_cable_lengths", data=cable_length_error)
                if gebf_cable_tensions.shape == result.cable_tensions.shape:
                    h5.create_dataset("gebf_cable_tensions", data=gebf_cable_tensions)
                    h5.create_dataset("pcs_minus_gebf_cable_tensions", data=result.cable_tensions - gebf_cable_tensions)

            fig_cmp, ax_cmp = plt.subplots(figsize=(8.4, 5.0))
            for j in range(result.cable_lengths.shape[0]):
                ax_cmp.plot(1.0e3 * progress, 1.0e3 * gebf_cable_lengths[j], "--", label=f"GEBF cable {j + 1}")
                ax_cmp.plot(1.0e3 * progress, 1.0e3 * result.cable_lengths[j], label=f"PCS cable {j + 1}")
            ax_cmp.set_xlabel("Path progress (mm)")
            ax_cmp.set_ylabel("Cable length (mm)")
            ax_cmp.grid(True)
            ax_cmp.legend(ncol=2)
            cable_length_compare_path = out_dir / f"tip_yz_{traj.name}_gebf_vs_pcs_cable_lengths.png"
            fig_cmp.savefig(cable_length_compare_path, dpi=220, bbox_inches="tight")

            fig_err, ax_err = plt.subplots(figsize=(8.4, 5.0))
            for j in range(cable_length_error.shape[0]):
                ax_err.plot(1.0e3 * progress, 1.0e3 * cable_length_error[j], label=f"cable {j + 1}")
            ax_err.set_xlabel("Path progress (mm)")
            ax_err.set_ylabel("PCS - GEBF cable length (mm)")
            ax_err.grid(True)
            ax_err.legend()
            cable_length_error_path = out_dir / f"tip_yz_{traj.name}_pcs_minus_gebf_cable_lengths.png"
            fig_err.savefig(cable_length_error_path, dpi=220, bbox_inches="tight")

            if gebf_cable_tensions.shape == result.cable_tensions.shape:
                fig_tau_cmp, ax_tau_cmp = plt.subplots(figsize=(8.4, 5.0))
                for j in range(result.cable_tensions.shape[0]):
                    ax_tau_cmp.plot(1.0e3 * progress, gebf_cable_tensions[j], "--", label=f"GEBF cable {j + 1}")
                    ax_tau_cmp.plot(1.0e3 * progress, result.cable_tensions[j], label=f"PCS cable {j + 1}")
                ax_tau_cmp.set_xlabel("Path progress (mm)")
                ax_tau_cmp.set_ylabel("Cable tension / multiplier")
                ax_tau_cmp.grid(True)
                ax_tau_cmp.legend(ncol=2)
                tension_compare_path = out_dir / f"tip_yz_{traj.name}_gebf_vs_pcs_tensions.png"
                fig_tau_cmp.savefig(tension_compare_path, dpi=220, bbox_inches="tight")
        else:
            print(
                "Skipping GEBF-vs-PCS cable comparison because cable_lengths shapes differ: "
                f"GEBF {gebf_cable_lengths.shape}, PCS {result.cable_lengths.shape}"
            )

        if gebf_solve_time_s is not None and gebf_solve_time_s.shape == result.solve_time_s.shape:
            solve_time_error = result.solve_time_s - gebf_solve_time_s
            with h5py.File(history_path, "a") as h5:
                h5.create_dataset("gebf_solve_time_s", data=gebf_solve_time_s)
                h5.create_dataset("pcs_minus_gebf_solve_time_s", data=solve_time_error)
                h5.attrs["gebf_solve_time_total_s"] = float(np.sum(gebf_solve_time_s))
                h5.attrs["pcs_solve_time_total_s"] = float(np.sum(result.solve_time_s))

            fig_time_cmp, ax_time_cmp = plt.subplots(figsize=(8.4, 5.0))
            ax_time_cmp.plot(1.0e3 * progress, gebf_solve_time_s, "--", label="GEBF")
            ax_time_cmp.plot(1.0e3 * progress, result.solve_time_s, label="PCS")
            ax_time_cmp.set_xlabel("Path progress (mm)")
            ax_time_cmp.set_ylabel("Solve time per target (s)")
            ax_time_cmp.grid(True)
            ax_time_cmp.legend()
            solve_time_compare_path = out_dir / f"tip_yz_{traj.name}_gebf_vs_pcs_solve_time.png"
            fig_time_cmp.savefig(solve_time_compare_path, dpi=220, bbox_inches="tight")

            fig_time_bar, ax_time_bar = plt.subplots(figsize=(5.6, 4.6))
            ax_time_bar.bar(["GEBF", "PCS"], [float(np.sum(gebf_solve_time_s)), float(np.sum(result.solve_time_s))])
            ax_time_bar.set_ylabel("Total solve time (s)")
            ax_time_bar.grid(True, axis="y")
            solve_time_total_compare_path = out_dir / f"tip_yz_{traj.name}_gebf_vs_pcs_total_solve_time.png"
            fig_time_bar.savefig(solve_time_total_compare_path, dpi=220, bbox_inches="tight")
        elif gebf_solve_time_s is None:
            print("GEBF history has no solve_time_s dataset; skipping solve-time comparison.")
        else:
            print(
                "Skipping solve-time comparison because solve_time_s shapes differ: "
                f"GEBF {gebf_solve_time_s.shape}, PCS {result.solve_time_s.shape}"
            )
    else:
        print(f"GEBF history file not found, skipping comparison: {gebf_history_path}")

    plt.close("all")

    errors = np.linalg.norm(result.tracking_error, axis=1)
    print("Trajectory:", TRAJECTORY)
    print("Gravity enabled:", bool(ENABLE_GRAVITY))
    print("Gravity vector:", gravity_vector)
    print("Solved points:", result.target_yz.shape[0])
    print("Successful optimizer points:", int(result.success.sum()), "/", result.success.shape[0])
    print("Max y-z tracking error (mm):", float(1.0e3 * errors.max()))
    print("Mean y-z tracking error (mm):", float(1.0e3 * errors.mean()))
    print("PCS solve time total (s):", float(np.sum(result.solve_time_s)))
    print("PCS solve time mean per target (s):", float(np.mean(result.solve_time_s)))
    print("PCS wall time total (s):", wall_time_total_s)
    if gebf_history_path.exists():
        print("GEBF solve time total from history (s):", gebf_solve_time_total_s)
    print("Cable length range (mm):")
    print(1.0e3 * np.column_stack([result.cable_lengths.min(axis=1), result.cable_lengths.max(axis=1)]))
    print("Cable tension range:")
    print(np.column_stack([result.cable_tensions.min(axis=1), result.cable_tensions.max(axis=1)]))
    print(f"Saved history to: {history_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved cable-length plot to: {length_plot_path}")
    print(f"Saved tension plot to: {tension_plot_path}")
    print(f"Saved path plot to: {path_plot_path}")
    print(f"Saved solve-time plot to: {time_plot_path}")
    if cable_length_compare_path is not None:
        print(f"Saved GEBF-vs-PCS cable-length comparison to: {cable_length_compare_path}")
    if cable_length_error_path is not None:
        print(f"Saved PCS-minus-GEBF cable-length plot to: {cable_length_error_path}")
    if tension_compare_path is not None:
        print(f"Saved GEBF-vs-PCS tension comparison to: {tension_compare_path}")
    if solve_time_compare_path is not None:
        print(f"Saved GEBF-vs-PCS solve-time comparison to: {solve_time_compare_path}")
    if solve_time_total_compare_path is not None:
        print(f"Saved GEBF-vs-PCS total solve-time comparison to: {solve_time_total_compare_path}")


if __name__ == "__main__":
    main()
