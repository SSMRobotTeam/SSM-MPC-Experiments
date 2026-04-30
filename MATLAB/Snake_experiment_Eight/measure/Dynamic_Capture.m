function Dynamic_Capture(luster,base_xyz,pos_endeff_ref,pos_mideff_ref,pos_feedbackPt_ref,R_B2G)
%DYNAMIC_CAPTURE 此处显示有关此函数的摘要
%   此处显示详细说明



% 传出   t_i   displ_endeff
% disp('Data receiving')
frameData = luster.receiveFrame();
% save time 
% frame_new = frameData.frameID;
% t_i       = (double(frame_new) - frame_ref) * dt;

% printFrameData(frameData);
rigid = frameData.allRigidbody;
markers   = frameData.allMarker;

%% 第二段末坐标
global mark_ind
global displ_endeff
endeff   = markers(mark_ind);
endeff_xyz   = [endeff.X, endeff.Y, endeff.Z];
delta_EB    = (endeff_xyz - base_xyz)'; % 末端执行器 与 基座坐标系的差值，在G系中表示
pos_endeff  = (R_B2G*delta_EB)';        % 转到基座坐标系下
displ_endeff = pos_endeff - pos_endeff_ref;

%% 第一段末坐标
global displ_mideff
mideff   = rigid(2);
mideff_xyz   = [mideff.X, mideff.Y, mideff.Z];
delta_MB    = (mideff_xyz - base_xyz)';   % 中端执行器 与 基座坐标系的差值，在G系中表示
pos_mideff  = (R_B2G*delta_MB)';        % 转到基座坐标系下
displ_mideff  = pos_mideff - pos_mideff_ref;        % 转到基座坐标系下

%% 第二段中间某点
global displ_feedbackPt
feedback_Pt       = rigid(3);
feedback_pt_xyz   = [feedback_Pt.X, feedback_Pt.Y, feedback_Pt.Z];
delta_FBptB       = (feedback_pt_xyz - base_xyz)'; % 中端 与 基座坐标系的差值，在G系中表示
pos_feedbackPt    = (R_B2G*delta_FBptB)';        % 转到基座坐标系下
displ_feedbackPt  = pos_feedbackPt - pos_feedbackPt_ref;        % 转到基座坐标系下


current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
timestamp = char(current_time);  % 将 datetime 转换为字符串

disp(['Receive Data @ ' timestamp]);

global y_exprm
% y_exprm = [y_exprm;...
%            [displ_endeff(1), displ_endeff(2), displ_endeff(3),...
%             displ_mideff(1), displ_mideff(2), displ_mideff(3)]
%             ];
y_exprm = [y_exprm;...
           [displ_endeff(1), displ_endeff(2), displ_endeff(3),...
            displ_mideff(1), displ_mideff(2), displ_mideff(3),...
            displ_feedbackPt(1), displ_feedbackPt(2), displ_feedbackPt(3)]
            ];

end

