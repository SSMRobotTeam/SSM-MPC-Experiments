% Identify the diamond robot's parameters with the original PSO settings.
run(fullfile(fileparts(mfilename('fullpath')), 'preprocess_data.m'));
import com.comsol.model.*
import com.comsol.model.util.*

filename_comsol = fullfile(project_dir, 'models', 'Diammond_old2.mph');
model = mphload(filename_comsol);

% Parameter order: Young's modulus, Poisson's ratio, alpha, beta.
num_vars = 4;
lb = [0.4e6, 0.45,  0.001, 0.001];
ub = [2e6,   0.499, 0.1,   0.05];

obj_fun = @(params) RMSE_Comsol_Exp(model, time, displ_endeff_save, ...
    params(1), params(2), params(3), params(4));

options = optimoptions('particleswarm', ...
    'SwarmSize', 15, ...
    'MaxIterations', 20, ...
    'HybridFcn', @fmincon);

[x, fval] = particleswarm(obj_fun, num_vars, lb, ub, options);
fprintf('Optimal parameters: [%.4f, %.4f, %.4f, %.4f]\n', x(1), x(2), x(3), x(4));
fprintf('Minimum objective value: %.4f\n', fval);
