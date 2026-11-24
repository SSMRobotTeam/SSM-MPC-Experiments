function RMSE = RMSE_Comsol_Exp(model,time,displ_endeff_save,E,nu,alpha,beta)
%RMSE_COMSOL_EXP 此处显示有关此函数的摘要
%   此处显示详细说明
%% 调整参数
model.component('comp1').material('mat1').propertyGroup('Enu').set('E', E);
model.component('comp1').material('mat1').propertyGroup('Enu').set('nu', nu);
% model.component('comp1').material('mat1').propertyGroup('def').set('density', {'1050'});
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM', alpha);
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK', beta);

model.component('comp1').physics('solid').feature('disp1').setIndex('Direction', 'free', 0);
model.component('comp1').physics('solid').feature('disp1').setIndex('Direction', 'free', 1);
model.component('comp1').physics('solid').feature('disp1').setIndex('Direction', 'free', 2);

model.study('std1').feature('stat').set('geometricNonlinearity', true);
model.study('std2').feature('time').set('geometricNonlinearity', true);
model.study("std1").run();
model.study('std2').feature('time').set('tlist', 'range(0,0.1,1)');
model.study("std2").run();

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

model.component('comp1').physics('solid').feature('disp1').setIndex('Direction', 'prescribed', 0);
model.component('comp1').physics('solid').feature('disp1').setIndex('Direction', 'prescribed', 1);
model.component('comp1').physics('solid').feature('disp1').setIndex('Direction', 'prescribed', 2);
model.component('comp1').physics('solid').feature('disp1').set('U0', {'14.3451e-3' ,'153.7418e-3' ,'-35.69e-3'});  
model.study("std1").run();

model.study('std2').feature('time').set('useinitsol', true);
model.study('std2').feature('time').set('initmethod', 'sol');
model.study('std2').feature('time').set('initstudy', 'std1');
model.study('std2').feature('time').set('solnum', 'last');
model.study('std2').feature('time').set('tlist', 'range(0,0.01,5)');

model.study("std2").run();
probe1= mphtable(model,"tbl1"); % comsol data

data_comsol = probe1.data(:, 2:end).*1000;
time_comsol = probe1.data(:, 1);


time_exp = time(1068:1568)-time(1068);
x_exp = displ_endeff_save(1068:1568,1);
y_exp = displ_endeff_save(1068:1568,2);
% endeff_disp_z = displ_endeff_save(1068:1568,3);

x_comsol = Endeff_x_interp(time_exp, time_comsol, data_comsol(:,1));
y_comsol = Endeff_y_interp(time_exp, time_comsol, data_comsol(:,2));

% 
squared_errors_x = (x_exp - x_comsol).^2;  % 误差的平方
MSE_x = mean(squared_errors_x);              % 均方误差（MSE）
RMSE_x = sqrt(MSE_x);                        % 均方根误差（RMSE）

squared_errors_y = (y_exp - y_comsol).^2;  % 误差的平方
MSE_y = mean(squared_errors_y);              % 均方误差（MSE）
RMSE_y = sqrt(MSE_y);                        % 均方根误差（RMSE）

% RMSE = RMSE_x + RMSE_y + RMSE_freq;
RMSE = RMSE_x + RMSE_y;
% RMSE = RMSE_y;

end

