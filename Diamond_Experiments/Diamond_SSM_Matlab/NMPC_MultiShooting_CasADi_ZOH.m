function [xt_opt, St_opt, ut_opt, J] = NMPC_MultiShooting_CasADi_ZOH(x0,x_guess,S0_guess,u_guess,uv_ref,cont,x_num,controlnum,Ad,Bd,dd,M,bhat,Hp,btilde)

%NMPCBYCASADI 此处显示有关此函数的摘要
% Xt : current state (reduced) Xt = [pk; qk]
% yref : reference trajectory
% Np : predictive horizon
% T : single step
% Dynamics : Dynamics equations
% cont : contains weights and other message
% linearmodes : linear modes number
% nonlinearmodes : nonlinear modes number
% controlnum : control inputs number

import casadi.*
% ------------------  Parameter introduction ---------------
Q  = cont.Q;
R  = cont.R;
P  = cont.P;

u_lb  = cont.u_lb;
u_ub  = cont.u_ub;

% Total control and States in Horizon
U         = SX.sym('U', controlnum, 1);
S0        = SX.sym('S0', x_num, 1);
X         = SX.sym('X',   x_num     , 2);
g         = [];  % equality and inequality equation

xk        = X(:,1); % initial state
g         = [g; xk-x0];             % x_num*1


xk   = X(:,1);  
xk1  = X(:,2);
S0k  = S0;
uk   = U;
% ZOH
xk1_calcu      = Ad * xk + Bd * S0k + dd;

x_residual     =   xk1 - xk1_calcu;
g              =  [g; x_residual];        % Continuation condition, x_num*1
cable_residual = Hp*xk1 + btilde + uk;
g              = [g; cable_residual];
yk             = M*xk1 + bhat;
obj            = (uv_ref - yk)' * Q * (uv_ref - yk) + S0k' * R * S0k + 0.01*(uk'*uk);

% ----------------------------------------------------------------
% -------------------- Struct NLP problem ------------------------
OPT_variables = [reshape(X, x_num*2     ,1);...
                 reshape(S0,x_num       ,1);...
                 reshape(U, controlnum  ,1)];    % [X0,X1,...,XNp, λ0,λ1,...,λNp-1, U0,U1,...,UNp-1]
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g);
options = struct;
options.ipopt.print_level = 0;%0,3
options.print_time = 0;
solver = nlpsol('solver', 'ipopt', nlp_prob, options);

args   = struct;

args.lbg   =  zeros(size(g));  % Equality constraints, g, state_guess - state_calculated = 0
args.ubg   =  zeros(size(g));  % constraints, 

args.lbx(1:x_num*3,1)   =  ones(x_num*3,1)*-inf; % lower and upper bound of states pt & S0t
args.ubx(1:x_num*3,1)   =  ones(x_num*3,1)*+inf; 

args.lbx(1+x_num*3:x_num*3+controlnum, 1) = u_lb; % lower and upper bound of control ut
args.ubx(1+x_num*3:x_num*3+controlnum, 1) = u_ub;


% ----------------------- Solver -------------------
args.x0 = [reshape(x_guess  ,x_num*2,1);...
           reshape(S0_guess ,x_num  ,1);...
           reshape(u_guess,        controlnum  ,1)];

% Initial values and Constraints
solution = solver('x0', args.x0,...
                  'lbx',args.lbx, 'ubx',args.ubx,...
                  'lbg',args.lbg, 'ubg',args.ubg);
J           = full(solution.f);
xt          = reshape(full(solution.x(1:x_num*2)), x_num, []);
xt_opt      = xt(:,2:end);
St_opt      = reshape(full(solution.x(1+x_num*2:x_num*3)), x_num, []);
ut_opt      = reshape(full(solution.x(1+x_num*3:x_num*3+controlnum)), controlnum, []);
end

