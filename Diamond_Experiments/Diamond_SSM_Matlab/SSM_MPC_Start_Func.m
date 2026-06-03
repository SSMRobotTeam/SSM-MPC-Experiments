function SSM_MPC_Start_Func
%SSM_MPC_START_FUNC 此处显示有关此函数的摘要
%   初始化 posi
disp('=================')
disp('Actuator Running')

global posi posi_hist zk_ssm veloc 
posi = [0,0,0,0];
posi_hist = posi;
veloc= [0,0,0,0]; % 拉绳速度
zk_ssm = [0,0,0];
end

