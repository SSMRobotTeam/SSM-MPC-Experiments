% Original V2_Lessmesh optimization settings; no diagnostic plotting sections.
run(fullfile(fileparts(mfilename('fullpath')), 'preprocess_data.m'));
import com.comsol.model.*
import com.comsol.model.util.*
mphopen -clear;
filename_comsol = fullfile(project_dir, 'models', 'trunk2_lessmesh.mph');
model = mphload(filename_comsol);

% ====================== 参数设置 ======================
% RMSE = RMSE_Comsol_Exp(model,time,displ_endeff_save,1e6,0.48,0.01,0.001);
% 参数包含：E,nu,alpha,beta
num_vars = 4;        % 待优化参数的数量（根据实际问题调整）
lb = [0.5e6,     0.47,  0.005,   0.0005];   % 参数下界（根据实际问题调整）
ub = [2e6,     0.499, 0.1,    0.01];    % 参数上界（根据实际问题调整）

obj_fun = @(params) RMSE_Comsol_Exp(model, time, displ_endeff_save, ...
                                    params(1), params(2), params(3), params(4));

% 创建优化选项
options = optimoptions('particleswarm', ...
    'SwarmSize', 15, ...                % 粒子群大小
    'MaxIterations', 30, ...           % 最大迭代次数
    'HybridFcn', @fmincon);          % 绘制最优解变化

% 运行粒子群算法
[x, fval] = particleswarm(obj_fun, num_vars, lb, ub, options);

% 显示结果
fprintf('优化结果:\n');
fprintf('最优解: x = [%.4f, %.4f, %.4f, %.4f]\n', x(1), x(2), x(3), x(4));
fprintf('目标函数最小值: fval = %.4f\n', fval);

