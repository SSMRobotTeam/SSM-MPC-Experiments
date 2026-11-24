# Free vibration calibration data and code

This package contains the experimental input and MATLAB calibration code for the trunk and diamond robots.

## Datasets

| Quantity | Trunk | Diamond |
| --- | --- | --- |
| Input file | [trunk CSV](trunk/data/free_vibration_input.csv) | [diamond CSV](diamond/data/free_vibration_input.csv) |
| Full recording samples | 4,342 | 1,588 |
| Recording timestamps | 0.010000 to 43.419998 s | 0.01 to 15.88 s |
| Recording span | 43.409998 s | 15.87 s |
| Nominal sampling frequency | 100 Hz | 100 Hz |
| Calibration rows (one-based, inclusive) | 1068–1568 | 1193–1250 |
| Calibration samples | 501 | 58 |
| Calibration timestamps | 10.68 to 15.68 s | 11.93 to 12.50 s |
| Calibration span | 5.00 s | 0.57 s |
| Displacement axes used for calibration | x and y | y and z |

Each CSV is a byte-for-byte copy of the input used by its local workflow. The trunk input comes from `V2/free.csv` and matches `V2_Lessmesh/free.csv`. The diamond input comes from `V2/free1.csv`. The upstream acquisition export for the trunk input has not been verified.

## Preprocessing and calibration

- **Trunk.** The 16 input columns contain the frame number, time, base quaternion and position, and end-effector quaternion and position. The CSV retains the original repeated axis headings. Preprocessing applies the original geometric offsets and coordinate transforms.
- **Diamond.** The 11 input columns contain time, four base quaternion components, base position, and end-effector position. Preprocessing uses the first frame as the reference and applies the original quaternion transform, 45° rotation, and axis sign changes.

Time is in seconds and positions and displacements are in millimetres. The scripts remove rows containing NaN without resampling. All rows in the supplied inputs are finite. Each calibration objective shifts its experimental window to start at zero, interpolates the COMSOL displacement at the experimental timestamps, and sums the RMSE for the two listed axes.

Both workflows fit Young's modulus, Poisson's ratio, and two Rayleigh damping coefficients with a swarm of 15 particles and `fmincon` as the hybrid solver. The trunk variants allow up to 30 iterations and differ in parameter bounds. The diamond workflow allows up to 20 iterations.

## Run in MATLAB

From the package root, run either preprocessing script to create `time` and `displ_endeff_save` in the workspace. Preprocessing requires `quat2dcm` from Aerospace Toolbox and runs independently of COMSOL.

```matlab
run('trunk/code/preprocess_data.m')
run('diamond/code/preprocess_data.m')
```

For parameter identification, use MATLAB with COMSOL LiveLink, Global Optimization Toolbox for `particleswarm`, and Optimization Toolbox for `fmincon`. Place the original COMSOL model at the path shown below before running the corresponding entry point.

| Workflow | MATLAB entry point | Model path |
| --- | --- | --- |
| Trunk V2 | [identify_parameters_v2.m](trunk/code/identify_parameters_v2.m) | `trunk/models/trunk2_v2.mph` |
| Trunk V2_Lessmesh | [identify_parameters_lessmesh.m](trunk/code/identify_parameters_lessmesh.m) | `trunk/models/trunk2_lessmesh.mph` |
| Diamond | [identify_parameters.m](diamond/code/identify_parameters.m) | `diamond/models/Diammond_old2.mph` |
