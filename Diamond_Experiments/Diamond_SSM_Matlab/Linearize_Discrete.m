function [Ad,Bd,dd,Mhat,bhat,Hp,btilde] = Linearize_Discrete(p_curr,z_curr,n,autData,Wmap,C_op,dt,Node_list)
%LINEARIZATION 此处显示有关此函数的摘要
% Local Jacobian of SSM by numerical method
%% Linearization of dynamics
% \dot{x} = Ax + Bu + d
R0       = R_auto_red_dyn(p_curr,autData);      % R(p_0)
R_p      = Rp_Jacob_V2(autData,p_curr,R0);       
W0       = R_reduced_to_full(p_curr,Wmap,[],0); % W(p_0)
W_p      = Wp_Jacob_V2(Wmap,p_curr,2*n,W0);      

% Btilde   = Utilde*Bext;
% verify jacobian
% [R_p_error, W_p_error] = Verify_Jacob(R_p,W_p,p_curr,autData,Wmap,2*n);
% fprintf("Jacobian matrix error  ε_R=%d,  ε_W=%d",R_p_error, W_p_error)

%% Linearize of SSM
% z = W_p*p + b
M  =   W_p;              Mz = M;
b  =  -W_p*p_curr + W0;  bz = b;

Mhat = C_op * M;  
bhat = C_op * b;

%% assemble
% \dot{x} = Ax + S0 + d
A = R_p;
B = eye(length(p_curr));
d = -R_p*p_curr + R0;

% invA = inv(A);
Ad = expm(dt*A);
Bd = A \ (Ad-eye(size(Ad)))*B;
dd = A \ (Ad-eye(size(Ad)))*d;

%% Linearization of cable constraints
% d(z) = Hp*p + btilde
% Hp   = Hz * z_p
d0 = CableLength_double(z_curr(1:n),Node_List);
Hz = Constraint_Hz(z_curr,Node_list);
Hp = Hz * zp;
btilde = -Hp*p_curr + d0;
end



