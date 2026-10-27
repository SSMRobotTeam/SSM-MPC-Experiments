# Free vibration calibration data and code

This package contains the preserved experimental input, preprocessing code, parameter-identification code, and required MATLAB helper functions. Model binaries, supplementary experiments, caches, and plot files are excluded.

## Dataset size and sampling

| Quantity | Value |
| --- | --- |
| Input file | `data/free_vibration_input.csv` |
| Input time points | 4342 |
| Timestamp range | 0.010000 to 43.419998 s |
| Recording span, last minus first | 43.409998 s |
| Nominal sampling frequency from timestamps | 100 Hz |
| Nominal sampling interval | 0.01 s |
| Calibration rows, one-based inclusive | 1068 through 1568 |
| Calibration time points | 501 |
| Calibration timestamp range | 10.68 to 15.68 s |
| Calibration duration | 5.0 s, approximately 0.08333 min |
| Compared displacement channels | x and y, 1002 scalar observations |

The input is copied byte-for-byte from the original `V2/free.csv`; it is also identical to `V2_Lessmesh/free.csv`. It is the preserved input used by the local identification workflow. Its upstream raw acquisition export has not been identified. None of the three nearby acquisition exports was established as its direct frame-wise source, so unrelated exports are excluded. Before claiming release of the original acquisition data, the author must supply or verify that source.

## Columns and preprocessing

Columns 1 to 16 are frame; time in seconds; base quaternion x,y,z,w; base position x,y,z in millimetres; end-effector quaternion x,y,z,w; and end-effector position x,y,z in millimetres. Repeated X/Y/Z/W headings are retained unchanged from the source CSV.

`preprocess_data.m` uses the original geometric offsets and transformations to calculate end-effector displacement relative to the initial reference. It removes rows containing NaN in the 16 selected columns; the supplied input contains 4342 finite rows, so no rows are removed. It preserves timestamps, performs no resampling, and creates `time` and `displ_endeff_save` in the workspace.

`RMSE_Comsol_Exp.m` selects rows 1068:1568 and subtracts the first selected timestamp. It interpolates COMSOL output at those experimental timestamps and minimizes the sum of x and y RMSE. The COMSOL study uses `range(0,0.01,5)`.

## Files and execution

- `code/preprocess_data.m`: read the input and compute displacement; no COMSOL model is needed for this step.
- `code/identify_parameters_v2.m`: V2 particle-swarm optimization settings.
- `code/identify_parameters_lessmesh.m`: V2_Lessmesh optimization settings.
- `code/Quat2Rot.m`, `code/RMSE_Comsol_Exp.m`, `code/Endeff_x_interp.m`, `code/Endeff_y_interp.m`: original helper functions.

In MATLAB, run `code/preprocess_data.m` to inspect the full displacement dataset. To identify parameters, use MATLAB with COMSOL LiveLink and the toolboxes providing `quat2dcm`, `particleswarm`, and `fmincon`. Supply the chosen original model separately as `models/trunk2_v2.mph` or `models/trunk2_lessmesh.mph`, then run the corresponding identification script. These model files are not included in this minimal data-and-code package; full optimization requires them. Exact software versions must be supplied by the author.

Both variants use the same input and objective. V2 uses a lower Young modulus bound of 1e6 and an upper alpha bound of 0.05; V2_Lessmesh uses 0.5e6 and 0.1. Both use four parameters, swarm size 15, maximum 30 iterations, and `fmincon` as the hybrid solver. The final manuscript variant remains for the author to confirm; neither is designated as final here.

Code was extracted from the original MLX code cells. Packaging removes MAT conversion caches, model-loading/plotting steps from preprocessing, and post-optimization diagnostic sections. The preprocessing calculations, objective function, and optimization settings are preserved. Paths are constructed relative to this package. MATLAB / COMSOL execution was not performed during packaging.

For the reviewer response, distinguish the complete input recording (4342 time points over 43.409998 s) from the actual calibration window (501 time points over 5 s). These figures describe the inspected local workflow; confirm its correspondence to the final manuscript before submission.
