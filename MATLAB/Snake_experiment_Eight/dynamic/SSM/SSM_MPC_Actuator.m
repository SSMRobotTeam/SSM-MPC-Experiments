function SSM_MPC_Actuator(cont,x_num,controlnum,auData,W_0_for_out,W_0_for_cables,W_0_for_FB,socket,feedback,options)
ssm_dir = fileparts(mfilename('fullpath'));
feedback_file = fullfile(ssm_dir, 'ssm_feedback.mat');
         
% tic
global posi  yk_ssm   factor
global posi_save
global displ_endeff  
global displ_mideff  
global displ_feedbackPt  
% tic
%% 驱动电机运行
% 获取当前时间（秒�?
current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
timestamp = char(current_time);  % �?datetime 转换为字符串
% 构造消息数�?
message = struct();
message.command = 'set_motors';
message.timestamp = timestamp;

message.motorsCount = 6;
message.isCtlModeUnified = 1;

message.param.ctlMode = [3,3,3,3,3,3];
message.param.posiSp  = [posi(1), posi(3), posi(2), posi(4), posi(6), posi(5)] * factor;

% message.param.velcSp  = veloc*factor;
message.param.velcSp  = [0,0,0,0,0,0];
message.param.forceSp = [0,0,0,0,0,0];

% 编码�?JSON 字符�?
jsonStr = jsonencode(message);

% 发送消息（带话题）
topic = '/controller/setActuatorDate';
socket.send(topic, org.zeromq.ZMQ.SNDMORE);  % 发送话�?
socket.send(jsonStr);  % 发送消�?

disp(['Sent @ ' message.timestamp]);
disp(['拉绳长度�?', num2str(posi)]);
% disp(['拉绳速度�?', num2str(veloc)]);

% toc
%% %%%%%%%%%%%%%%%%%%%%  SSM-MPC 计算当前控制�? %%%%%%%%%%%%%%%%%%%% 

% disp('SSM-MPC running')
%SSM_MPC 此处显示有关此函数的摘要
load(feedback_file,'ssm_feedback')
t         = ssm_feedback.t;     % 当前时间
pk        = ssm_feedback.pk;    % 上一时刻的ssm状态量
k         = ssm_feedback.k;     % 时步
S0_opt    = ssm_feedback.S0_opt;% S0最优值，由上一步计算得到，用于下一步的猜测
ut_opt    = ssm_feedback.ut_opt;% ut最优值，由上一步计算得到，用于下一步的猜测

tic
%% 测量反馈, 可以单独写成函数
if feedback == 1 
% 测试 - 随机扰动
    % zk_fb_ssm   =   R_reduced_to_full(pk,W_0_for_FB,[],1);
    % zk_meas     =   zk_fb_ssm.* (1 + (rand(9,1)-0.5)*2*0.05);
% 真实测量
    zk_meas = [displ_endeff(1), displ_endeff(2), displ_endeff(3),...
               displ_mideff(1), displ_mideff(2), displ_mideff(3),...
               displ_feedbackPt(1), displ_feedbackPt(2), displ_feedbackPt(3)
               ]' ./ 1000;

%     if norm(zk_meas(1:3), 2) > 0.025
        
%         [pk_fb, residual_k, residuals]  =  lsqnonlin(@(p) Proj_err(zk_meas, p, W_0_for_FB, opt_W, pk1), pk1, [], [], options);
        [pk_fb, ~, residuals]  =  lsqnonlin(@(p) Func_Feedback_pk(double(zk_meas), p, W_0_for_FB, pk, cont.opt_W), pk, [], [], options);
        residual_norm = norm(residuals(1:9), 2);
        if residual_norm < 1e-2
            % 更新状态变量，保存反馈情况
            pk       = pk_fb;
        end
%     end
end

% calculate current states
zk_outdof       = R_reduced_to_full(pk,W_0_for_out,    [],1);
zk_cables       = R_reduced_to_full(pk,W_0_for_cables ,[],1);

% warm-start
p_guess  = [pk, pk];
S0_guess = [S0_opt(:,2:end), S0_opt(:,end)];
u_guess  = [ut_opt(:,2:end), ut_opt(:,end)];

% 提取预期轨迹
ref_uv = [];
for i = 1:cont.Np
    t_i = t + (i-1)*cont.dt;
    if t_i + (cont.Np-1)*cont.dt >= cont.tf
        yref     = interp1(cont.td, cont.zd',t_i        ,'spline','extrap')';
        yref_1   = interp1(cont.td, cont.zd',t_i+cont.dt,'spline','extrap')';
        dyref    = (yref_1 - yref) / cont.dt;
    else
        yref     = cont.zd(:,k+i-1);
        dyref    = (cont.zd(:,k+i) - cont.zd(:,k+i-1)) / cont.dt;
    end
    ref_uv = [ref_uv, [yref; dyref]];
end

Node_List = cont.Node_List;
% Linearize SSM dynamics 
R0       = R_auto_red_dyn(pk,auData);      % R(p_0)
R_p      = Rp_Jacob_V2(auData,pk,R0);      

%% assemble - discrete
% \dot{x} = Ax + S0 + d
A = R_p;
B = eye(length(pk));
d = -R_p*pk + R0;

Ad = expm(cont.dt*A);
Bd = A \ (Ad-eye(size(Ad)))*B;
dd = A \ (Ad-eye(size(Ad)))*d;

pk1   = pk;

% iteration for Bext
% tic

% Linearize W map
% z = W_p*p + b
W0_outdofs       = zk_outdof(:,1); % W(p_0)
W0_cables        = zk_cables(:,1); % W(p_0)

W_p_outdofs      = Wp_Jacob_V2(W_0_for_out,pk,size(W_0_for_out(1).coeffs,1),W0_outdofs);
W_p_cables       = Wp_Jacob_V2(W_0_for_cables,pk,size(W_0_for_cables(1).coeffs,1),W0_cables);

M  =   W_p_outdofs;              
b  =  -W_p_outdofs*pk + W0_outdofs;  

Mhat = [M(1:3,:); M(7:9,:)];    % 提取末端坐标 + 速度
bhat = [b(1:3,:); b(7:9,:)];


% Linearize cable constraints
d0 = CableLength_double_part(zk_cables,Node_List);
Hz = Constraint_Hz_part(zk_cables,Node_List);

Hp = Hz * W_p_cables;
btilde = -Hp*pk + d0 - cont.dist0;
% 

% d0 = CableLength_double(zk(1:cont.n),Node_List);
% Hz = Constraint_Hz(zk,Node_List);
% Hp = Hz * W_p;
% btilde = -Hp*pk1 + d0 - cont.dist0;

% OPtimization
[pt_opt, S0_opt, ut_opt, ~] = NMPC_MultiShooting_CasADi_ZOH(pk,p_guess,S0_guess,u_guess,ref_uv,cont,x_num,controlnum,Ad,Bd,dd,Mhat,bhat,Hp,btilde);
% S01   = St_opt(:,1);     % only take the first Nc control input
pk1   = pt_opt(:,1);
uk1   = ut_opt(:,1);
pk    =  pk1;

% disp(['位移计算�? ', num2str(zk(cont.outdof)'.*1000)]);
% disp(['位移测量�? ', num2str(displ_endeff)]);

% 控制异常检�?
if ~sum(isnan(uk1))
    % 拉绳长度和速度
    posi     = uk1'*1000;
    % 保护代码,防止电机拉伸过大
    if sum(abs(posi.*factor) > 100) > 0
        posi = posi * 0;
    end
    yk_ssm   = zk_outdof(1:6)'*1000; % 提取末端点和中间端点的空间坐�?
end

% save new ssm result for next loop
ssm_feedback.t        = t+cont.dt;
ssm_feedback.k        = k + 1;
ssm_feedback.pk       = pk;
ssm_feedback.S0_opt  = S0_opt;
ssm_feedback.ut_opt  = ut_opt;
save(feedback_file,'ssm_feedback')

posi_save = [posi_save; posi];

toc
end

