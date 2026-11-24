function RMSE = RMSE_Comsol_Exp(model,time,displ_endeff_save,E,nu,alpha,beta)
%RMSE_COMSOL_EXP 此处显示有关此函数的摘要
%   此处显示详细说明
%% 调整参数
model.component('comp1').material('mat2').propertyGroup('Enu').set('E', E);
model.component('comp1').material('mat2').propertyGroup('Enu').set('nu', nu);
% model.component('comp1').material('mat2').propertyGroup('def').set('density', {'1050'});
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM', alpha);
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK', beta);

model.study('std1').feature('stat').set('geometricNonlinearity', true);
model.study('std2').feature('time').set('geometricNonlinearity', true);

% MA = mphmatrix(model ,'sol2', ...
%  'Out', {'K','D','E','Null','Nullf','ud','uscale'},'initmethod','sol','initsol','zero');
% M_0 = MA.E; C_0 = MA.D; K_0 = MA.K; Null = MA.Null;
% Nullf = MA.Nullf; 
% M = Nullf'*M_0*Null;
% C = Nullf'*C_0*Null;
% K = Nullf'*K_0*Null;

%% 固有频率
% [~,omega2] = eigs(K,M,2,'SM');
% omega = sqrt(diag(omega2));
% Freq_comsol = omega/(2*pi)

model.component('comp1').physics('solid').feature('disp5').setIndex('Direction', 'prescribed', 0);
model.component('comp1').physics('solid').feature('disp5').setIndex('Direction', 'prescribed', 1);
model.component('comp1').physics('solid').feature('disp5').setIndex('Direction', 'prescribed', 2);
model.component('comp1').physics('solid').feature('disp5').set('U0', {'0' ,'-16.7242e-3' ,'-8.2476e-3'});  
% model.component('comp1').physics('solid').feature('disp5').set('U0', {'0' ,'-11.5783e-3' ,'-5.4e-3'});  
model.study("std1").run();

model.component('comp1').physics('solid').feature('disp5').setIndex('Direction', 'free', 0);
model.component('comp1').physics('solid').feature('disp5').setIndex('Direction', 'free', 1);
model.component('comp1').physics('solid').feature('disp5').setIndex('Direction', 'free', 2);

model.study('std2').feature('time').set('useinitsol', true);
model.study('std2').feature('time').set('initmethod', 'sol');
model.study('std2').feature('time').set('initstudy', 'std1');
model.study('std2').feature('time').set('solnum', 'last');
model.study('std2').feature('time').set('tlist', 'range(0,0.01,0.57)');

model.study("std2").run();
probe1= mphtable(model,"tbl1"); % comsol data

data_comsol = probe1.data(:, 2:end).*1000;
time_comsol = probe1.data(:, 1);


time_exp = time(1193:1250)-time(1193);
y_exp = displ_endeff_save(1193:1250,2);
z_exp = displ_endeff_save(1193:1250,3);

y_comsol = Endeff_x_interp(time_exp, time_comsol, data_comsol(:,2));
z_comsol = Endeff_y_interp(time_exp, time_comsol, data_comsol(:,3));

% 
squared_errors_y = (y_exp - y_comsol).^2;  % 误差的平方
MSE_y = mean(squared_errors_y);              % 均方误差（MSE）
RMSE_y = sqrt(MSE_y);                        % 均方根误差（RMSE）

squared_errors_z = (z_exp - z_comsol).^2;  % 误差的平方
MSE_z = mean(squared_errors_z);              % 均方误差（MSE）
RMSE_z = sqrt(MSE_z);                        % 均方根误差（RMSE）

RMSE = RMSE_y + RMSE_z;

end

