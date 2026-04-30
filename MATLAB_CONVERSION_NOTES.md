# MATLAB to Python Conversion Notes

The project-specific MATLAB files from `MATLAB/Snake_experiment_Eight` have been ported into the current Python program tree.

- SSM controller code: `controller/`
- SSM model algorithms and data: `dynamics/SSM/`
- SSM ROM data: `dynamics/SSM/ROM8_Expe/`
- SSM main entry: `Main_demo_SSM.py`
- SSM measurement helpers: `measure/ssm_measurement.py`
- SSM plots: `PlotFig/plot_ssm_results.py`
- Trajectory helpers: `utils/trajectory_utils.py`

The MATLAB `casadi/` folder is a third-party MATLAB distribution and was not line-by-line converted. The Python NMPC port uses the Python `casadi` package instead.

MATLAB `.mlx` live scripts and `.slx` Simulink models are preserved as references under `MATLAB/Snake_experiment_Eight`; their executable logic has been represented by Python modules where practical.
