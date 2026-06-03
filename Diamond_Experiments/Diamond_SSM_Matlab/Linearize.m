function [A,B,d,M,bhat,Bhat,Btilde] = Linearize(x_curr,z_curr,np,nq,n,autData,Wmap,Lamhat_ext,Vhat_trans,G_ext,H_ext,C_op,Node_list)
%LINEARIZATION 此处显示有关此函数的摘要
%   此处显示详细说明
% Local Jacobian of SSM by numerical method
p_curr   = x_curr(1:np);
% q_curr   = x_curr(1+np:end);
% x = [p; q]
Bext     = -Constraint_Hz(zeros(n,1),Node_list);
% Bext     = Control_Mat_H(z_curr(1:n),Node_list);
Bhat     = H_ext*Bext;
Btilde   = G_ext*Bext; % control matrix of nonlin dy

%% Linearization of dynamics
% \dot{x} = Ax + Bu + d
R0       = R_auto_red_dyn(p_curr,autData);      % R(p_0)
R_p      = Rp_Jacob_V2(autData,p_curr,R0);       
% R_p     = Rp_Jacobian_ROM6(p_curr, autData);
W0       = R_reduced_to_full(p_curr,Wmap,[],0); % W(p_0)
W_p      = Wp_Jacob_V2(Wmap,p_curr,2*n,W0);      
% W_p      = Wp_Jacobian_ROM6(Wmap,p_curr);

% R0       = C_auto_red_dyn(p_curr,autData);      % R(p_0)
% R_p      = Rp_Jacob_V2(autData,p_curr,R0);       
% % R_p     = Rp_Jacobian_ROM6(p_curr, autData);
% W0       = C_reduced_to_full(p_curr,Wmap,[],0); % W(p_0)
% W_p      = Wp_Jacob_V2(Wmap,p_curr,2*n,W0);      
% % W_p      = Wp_Jacobian_ROM6(Wmap,p_curr);

% verify jacobian
% [R_p_error, W_p_error] = Verify_Jacob(R_p,W_p,p_curr,autData,Wmap,2*n);
% fprintf("Jacobian matrix error  ε_R=%d,  ε_W=%d",R_p_error, W_p_error)

%% assemble
A = blkdiag(R_p, Lamhat_ext);
B = [Btilde ; Bhat];
d = [-R_p*p_curr + R0;   zeros(nq,1)];

%% Linearize of SSM
% z = Hx + b
H  =  [W_p, Vhat_trans];
b  =  -W_p*p_curr + W0;

M    = C_op * H;
bhat = C_op*b;
end

