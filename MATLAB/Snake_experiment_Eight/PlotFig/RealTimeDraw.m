function RealTimeDraw
%REALTIMEDRAW 此处显示有关此函数的摘要
% 绘制 测量数据 
global hFig
global t_i  displ_endeff  displ_endeff_history zzd 
global zk_ssm zk_ssm_history 
global y_exprm

% disp(['测量值= ', num2str(displ_endeff)]);
% disp(['计算值= ', num2str(zk_ssm)]);

% % % % % 首次绘图时创建图形窗口（避免后台线程隐藏窗口）
% % % if ~exist('hFig', 'var') || isempty(hFig)  % hFig 为全局句柄
% % %     hFig = figure('Name', '实时绘图窗口', 'Visible', 'on');  % 强制窗口可见
% % % end
% % % figure(hFig);  % 激活窗口
% % % %   实时绘制测量数据的图像
% % % % scatter(displ_endeff(1), displ_endeff(3), 'bx');
% % % plot(zzd(1,:), zzd(2,:), 'k:', 'LineWidth', 3);
% % % hold on;
% % % % plot([zk_ssm_history(1), zk_ssm(1)], [zk_ssm_history(2), zk_ssm(2)], 'b-', 'LineWidth', 2);
% % % % hold on;
% % % plot([displ_endeff_history(1), displ_endeff(1)], [displ_endeff_history(2), displ_endeff(2)], 'r-.', 'LineWidth', 3);
% % % hold on;
% % % 
% % % legend('desired','measured', 'FontSize', 14, 'Location', 'best')
% % % % legend('ssm-mpc', 'measured', 'desired')
% % % xlabel('x/mm',FontSize=14)
% % % ylabel('y/mm',FontSize=14)
% % % 
% % % grid on;
% % % drawnow;

displ_endeff_history = displ_endeff;
zk_ssm_history       = zk_ssm;

y_exprm = [y_exprm; [-displ_endeff(1), displ_endeff(2), displ_endeff(3)]];


end

