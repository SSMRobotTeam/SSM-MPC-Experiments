function RealTime_Draw_Start
%REALTIME_DRAW_START 此处显示有关此函数的摘要
%   此处显示详细说明
global t_i displ_endeff displ_endeff_history 
global zk_ssm_history zk_ssm
t_i = 0;
displ_endeff         = [0,0,0];
displ_endeff_history = [0,0,0];

zk_ssm               = [0,0,0];
zk_ssm_history       = [0,0,0];
figure('Name', '实时绘图窗口');  % 初始化图形窗口

end

