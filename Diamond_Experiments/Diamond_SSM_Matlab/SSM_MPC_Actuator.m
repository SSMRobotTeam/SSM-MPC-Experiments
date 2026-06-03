function SSM_MPC_Actuator(zd, cont,x_num,controlnum,auData,W_0,socket,feedback,Uforced,BB)
         
% tic 

global posi zk_ssm veloc posi_hist displ_endeff  displ_endeff_hist factor
% tic
%% 驱动电机运行
% 获取当前时间（秒）
current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
timestamp = char(current_time);  % 将 datetime 转换为字符串
% 构造消息数据
message = struct();
message.command = 'set_motors';
message.timestamp = timestamp;

message.motorsCount = 5;
message.isCtlModeUnified = 1;

message.param.ctlMode = [3,3,3,3,3];
% message.param.posiSp  = [0,posi(3),posi(4),posi(1),posi(2)]*factor;
% message.param.posiSp  = [0,posi(3)+3,posi(4)+3,posi(1)+3,posi(2)+3]*factor;
% message.param.posiSp  = [0,posi(3), posi(4), posi(1), posi(2)]*factor;
message.param.posiSp  = [0,posi(3)+0.65, posi(4)+0.65, posi(1)+0.65, posi(2)+0.65]*factor;
message.param.velcSp  = [0,veloc(3),veloc(4),veloc(1),veloc(2)]*factor;
message.param.forceSp = [0,0,0,0,0];

% 编码为 JSON 字符串
jsonStr = jsonencode(message);

% 发送消息（带话题）
topic = '/controller/setActuatorDate';
socket.send(topic, org.zeromq.ZMQ.SNDMORE);  % 发送话题
socket.send(jsonStr);  % 发送消息

disp(['Sent @ ' message.timestamp]);
disp(['拉绳长度： ', num2str(posi)]);
% disp(['拉绳速度： ', num2str(veloc)]);

% toc
%% %%%%%%%%%%%%%%%%%%%%  SSM-MPC 计算当前控制量  %%%%%%%%%%%%%%%%%%%% 
tic
% disp('SSM-MPC running')
%SSM_MPC 此处显示有关此函数的摘要
load('ssm_feedback.mat','ssm_feedback')
t        = ssm_feedback.t;
pk       = ssm_feedback.pk;
zk       = ssm_feedback.zk;
S0_guess = ssm_feedback.S0_guess;
p_guess  = ssm_feedback.p_guess;
u_guess  = ssm_feedback.u_guess;

% load('ssm_save.mat', 'ssm_save')
% pt_save  =  ssm_save.pt_save;
% zt_save  =  ssm_save.zt_save;
% ztfull_save = ssm_save.ztfull_save;
% ut_save  =  ssm_save.ut_save;

yref  = zd(t);
dyref  = (zd(t+cont.dt)-zd(t)) / cont.dt;
uv_ref = [yref; dyref];

C_op = cont.C_op;
Node_List = cont.Node_list;
% Linearize SSM dynamics 
R0       = R_auto_red_dyn(pk,auData);      % R(p_0)
R_p      = Rp_Jacob_V2(auData,pk,R0);      

%% assemble
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
for jj = 1:1
% Linearize W map
    % z = W_p*p + b
    W0       = R_reduced_to_full(pk1,W_0,[],0); % W(p_0)
    W_p      = Wp_Jacob_V2(W_0,pk1,2*cont.n,W0);  

    M  =   W_p;              
    b  =  -W_p*pk1 + W0;  
    
    Mhat = C_op * M;  
    bhat = C_op * b;

% Linearize cable constraints
    d0 = Cable_length(zk(1:cont.n),  Node_List);
    Hz = Hz_constraint(zk(1:cont.n), Node_List);
    Hp = Hz * W_p;
    btilde = -Hp*pk1 + d0 - cont.dist0;

% OPtimization
    [pt_opt, St_opt, ut_opt, ~] = NMPC_MultiShooting_CasADi_ZOH(pk,p_guess,S0_guess,u_guess,uv_ref,cont,x_num,controlnum,Ad,Bd,dd,Mhat,bhat,Hp,btilde);
    S01   = St_opt;     % only take the first Nc control input
    pk1   = pt_opt;
    uk1   = ut_opt;
% calculate current states
    zk       = R_reduced_to_full(pk1,W_0,[],1);
    p_guess  = [pk1, pk1];
    S0_guess = S01;
    u_guess  = uk1;
end
pk  =  pk1;

% disp(['位移计算值= ', num2str(zk(cont.outdof)'.*1000)]);
disp(['预期位移  = ', num2str(yref'.*1000)]);
% disp(['位移测量值= ', num2str(displ_endeff)]);
%% 测量反馈
if feedback == 1 
    zk_feedback   = zk;
    zk_feedback(cont.outdof) = [ displ_endeff(1)./1000;...
                                 displ_endeff(2)./1000;...
                                 displ_endeff(3)./1000];
    veloc_endeff = (displ_endeff - displ_endeff_hist) / cont.dt;
    zk_feedback(cont.outdof+cont.n) = [ veloc_endeff(1)./1000;...
                                        veloc_endeff(2)./1000;...
                                        veloc_endeff(3)./1000];
    % disp(['速度= ', num2str(veloc_endeff),'mm']);
    pk_feedback   = Uforced*BB*zk_feedback; % proj to reduced space
    p_guess  = [pk_feedback, pk_feedback];
else
    zk_feedback = zk;
    pk_feedback = pk;
end


% 控制异常检测
if ~sum(isnan(uk1))
    posi     = uk1'*1000;
    veloc    = (posi-posi_hist)/cont.dt;
    zk_ssm   = zk(cont.outdof)'*1000;
end
posi_hist = posi;
displ_endeff_hist = displ_endeff;
% save new ssm result for next loop
ssm_feedback.t        = t+cont.dt;
ssm_feedback.pk       = pk_feedback;
ssm_feedback.zk       = zk_feedback;
ssm_feedback.S0_guess = S0_guess;
ssm_feedback.p_guess  = p_guess;
ssm_feedback.u_guess  = u_guess;
save('ssm_feedback.mat','ssm_feedback')

% 
% % save result
% ssm_save.pt_save  =  [pt_save, pk1];
% ssm_save.zt_save  =  [zt_save, zk(cont.outdof)];
% ssm_save.ztfull_save = [ztfull_save, zk];
% ssm_save.ut_save  =  [ut_save, uk1];
% save('ssm_save.mat','ssm_save')

toc
end

