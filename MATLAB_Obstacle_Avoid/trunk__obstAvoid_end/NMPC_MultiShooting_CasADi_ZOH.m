function [xt_opt, St_opt, ut_opt, J] = NMPC_MultiShooting_CasADi_ZOH(x0,x_guess,S0_guess,u_guess,ref_uv,cont,x_num,controlnum,Ad,Bd,dd,Mhat,bhat,Hp,btilde)

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
    Np = cont.Np;
    
    u_lb  = cont.u_lb;
    u_ub  = cont.u_ub;
    
    % Total control and States in Horizon
    U         = SX.sym('U', controlnum, Np);
    S0        = SX.sym('S0', x_num, Np);
    X         = SX.sym('X',   x_num     , Np+1);
    g         = [];  % equality and inequality equation
    
    xk        = X(:,1); % initial state
    g         = [g; xk-x0];             % x_num*1
    
    obj       = 0;
    delta_u   = [];
    delta_S0   = [];

    delta_u = [delta_u, U(:,1) - u_guess(:,1)];
    delta_S0 = [delta_S0, S0(:,1) - S0_guess(:,1)];
    for i = 1:Np-1
        delta_u = [delta_u, U(:,i+1)-U(:,i)];
        delta_S0 = [delta_S0, S0(:,i+1)-S0(:,i)];
    end


    for i = 1:Np
        xk   = X(:,i);  
        xk1  = X(:,i+1);
        S0k  = S0(:,i);
        uk   = U(:,i);
        
        % ZOH
        xk1_calcu      = Ad * xk + Bd * S0k + dd;
        
        x_residual     =   xk1 - xk1_calcu;
        g              =  [g; x_residual];        % Continuation condition, x_num*Np
        cable_residual = Hp*xk1 + btilde + uk;
        g              = [g; cable_residual];

        yk             = Mhat*xk1 + bhat;

        obj            = obj + (ref_uv(:,i) - yk)' * Q * (ref_uv(:,i) - yk)...
                             + 1e-1*R*(S0k'*S0k) + 1e0*R*(delta_S0(:,i)'*delta_S0(:,i))...
                             + 1e0*R*(uk'*uk)    + 5e3*R*(delta_u(:,i)'*delta_u(:,i));
    end
    
    % ----------------------------------------------------------------
    % -------------------- Struct NLP problem ------------------------
    OPT_variables = [reshape(X, x_num*(Np+1)     ,1);...
                     reshape(S0,x_num*Np       ,1);...
                     reshape(U, controlnum*Np  ,1)];    % [X0,X1,...,XNp, λ0,λ1,...,λNp-1, U0,U1,...,UNp-1]
    len_g  = length(g);
    
    nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g);
    options = struct;
    options.ipopt.print_level = 0;%0,3
    options.print_time = 0;

    solver = nlpsol('solver', 'ipopt', nlp_prob, options);
    
    args   = struct;
    args.lbg(1:len_g,1)   =  zeros(size(g));  % Equality constraints, g, state_guess - state_calculated = 0
    args.ubg(1:len_g,1)   =  zeros(size(g));  % constraints, 
    args.lbx(1:x_num*(2*Np+1),1)   =  ones(x_num*(2*Np+1),1)*-inf; % lower and upper bound of states pt & S0t
    args.ubx(1:x_num*(2*Np+1),1)   =  ones(x_num*(2*Np+1),1)*+inf; 
    args.lbx(1+x_num*(2*Np+1):x_num*(2*Np+1)+controlnum*Np, 1) = u_lb; % lower and upper bound of control ut
    args.ubx(1+x_num*(2*Np+1):x_num*(2*Np+1)+controlnum*Np, 1) = u_ub;
    
    
    % ----------------------- Solver -------------------
    args.x0 = [reshape(x_guess  ,x_num*(Np+1),1);...
               reshape(S0_guess ,x_num*Np  ,1);...
               reshape(u_guess,  controlnum*Np  ,1)];
    
    % Initial values and Constraints
    solution = solver('x0', args.x0,...
                      'lbx',args.lbx, 'ubx',args.ubx,...
                      'lbg',args.lbg, 'ubg',args.ubg);
    J           = full(solution.f);
    xt          = reshape(full(solution.x(1:x_num*(Np+1))), x_num, []);
    xt_opt      = xt(:,2:end);
    St_opt      = reshape(full(solution.x(1+x_num*(Np+1):x_num*(2*Np+1))), x_num, []);
    ut_opt      = reshape(full(solution.x(1+x_num*(2*Np+1):end)), controlnum, []);
end

