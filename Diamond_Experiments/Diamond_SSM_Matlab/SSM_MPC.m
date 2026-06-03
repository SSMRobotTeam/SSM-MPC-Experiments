function SSM_MPC(zd, cont,x_num,controlnum,auData,W_0)
% tic 
disp('Actuator Running')
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
C_op = cont.C_op;
Node_List = cont.Node_List;
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
tic
for jj = 1:2
% Linearize W map
    % z = W_p*p + b
    W0       = R_reduced_to_full(pk1,W_0,[],0); % W(p_0)
    W_p      = Wp_Jacob_V2(W_0,pk1,2*cont.n,W0);  

    M  =   W_p;              
    b  =  -W_p*pk1 + W0;  
    
    Mhat = C_op * M;  
    bhat = C_op * b;

% Linearize cable constraints
    d0 = CableLength_double(zk(1:cont.n),Node_List);
    Hz = Constraint_Hz(zk,Node_List);
    Hp = Hz * W_p;
    btilde = -Hp*pk1 + d0 - cont.dist0;

% OPtimization
    [pt_opt, St_opt, ut_opt, ~] = NMPC_MultiShooting_CasADi_ZOH(pk,p_guess,S0_guess,u_guess,yref,cont,x_num,controlnum,Ad,Bd,dd,Mhat,bhat,Hp,btilde);
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

global posi
posi     = uk1'*1000;

% save new ssm result for next loop
ssm_feedback.t        = t+cont.dt;
ssm_feedback.pk       = pk;
ssm_feedback.zk       = zk;
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

% toc
end

