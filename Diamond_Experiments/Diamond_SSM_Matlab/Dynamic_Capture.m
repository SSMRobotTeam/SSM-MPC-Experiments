function Dynamic_Capture(luster,frame_ref,dt,base_xyz,endeff_at_base_ref,R_BG,R_comsol)
%DYNAMIC_CAPTURE 此处显示有关此函数的摘要
%   此处显示详细说明
global t_i displ_endeff 
global y_exprm
% 传出   t_i   displ_endeff
% disp('Data receiving')
frameData = luster.receiveFrame();
% save time 
frame_new = frameData.frameID;
t_i       = (double(frame_new) - frame_ref) * dt;

endeff_data  = frameData.allMarker;

global end_index
% end_index = 3;
endeff       = endeff_data(end_index);

endeff_xyz   = [endeff.X, endeff.Y, endeff.Z];

delta_EB       = (endeff_xyz - base_xyz)'; % 末端执行器 与 基座坐标系的差值，在G系中表示
endeff_at_base = R_BG*delta_EB;

displ_endeff = R_comsol*(endeff_at_base - endeff_at_base_ref);
displ_endeff(1) = -displ_endeff(1);
displ_endeff(2) = -displ_endeff(2);
disp(['位移测量值= ', num2str(displ_endeff')]);

current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
timestamp = char(current_time);  % 将 datetime 转换为字符串
disp(['Receive Data @ ' timestamp]);
% disp(['t_i= ', num2str(t_i)]);

y_exprm = [y_exprm; [displ_endeff(1), displ_endeff(2), displ_endeff(3)]];
end

