% Transform the recorded end-effector position into COMSOL coordinates.
% Outputs time in seconds and displ_endeff_save in millimetres.
code_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(code_dir);
csv_file = fullfile(project_dir, 'data', 'free_vibration_input.csv');
recorded_data = readtable(csv_file);
free_mat = table2array(recorded_data(:, 1:11));
free_mat = free_mat(~any(isnan(free_mat), 2), :);

theta = pi/4;
R_comsol = [cos(theta),  sin(theta), 0; ...
           -sin(theta),  cos(theta), 0; ...
            0,           0,          1];

time = free_mat(:, 1);
base_xyz = free_mat(1, 6:8);
base_q = free_mat(1, 2:5);
R_BG = quat2dcm(base_q);
endeff_at_base_ref = R_BG * (free_mat(1, 9:11) - base_xyz)';

num_samples = size(free_mat, 1);
displ_endeff_save = zeros(num_samples, 3);
for i = 1:num_samples
    delta_EB = (free_mat(i, 9:11) - base_xyz)';
    endeff_at_base = R_BG * delta_EB;
    displ_endeff_save(i, :) = (R_comsol * (endeff_at_base - endeff_at_base_ref))';
end
displ_endeff_save(:, 1:2) = -displ_endeff_save(:, 1:2);
