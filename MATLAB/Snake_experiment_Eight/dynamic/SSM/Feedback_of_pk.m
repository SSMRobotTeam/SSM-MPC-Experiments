function pk_opt = Feedback_of_pk(pk, zk_meas, W_0_for_out)
    
    import casadi.*
    % ------------------  Parameter introduction ---------------
    x_num     = length(pk);
    X         = SX.sym('X',   x_num     , 1);
    g         = [];  % equality and inequality equation

    zk_guess  = R_reduced_to_full(X, W_0_for_out, [], 1);

    % yk             = Mhat*xk1 + bhat;
    obj            = 0.5 * (zk_guess(1:6)-zk_meas)'*(zk_guess(1:6)-zk_meas);

    
    % ----------------------------------------------------------------
    % -------------------- Struct NLP problem ------------------------
    OPT_variables = X;    % [X0,X1,...,XNp, λ0,λ1,...,λNp-1, U0,U1,...,UNp-1]
    len_g  = length(g);
    
    nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g);
    options = struct;
    options.ipopt.print_level = 0;%0,3
    options.print_time = 0;
    % options.compiler = 'mex';  % 编译为MEX文件
    % options.jit = true;        % 启用JIT加速
    
    % make sure the computing time short enough
    % options.ipopt.max_iter = 3;
    % options.ipopt.tol = 1e-2;
    % options.ipopt.linear_solver = 'wsmp';
    % options.ipopt.max_cpu_time = 4e-3;  % 最大计算时间（秒）
    % options.ipopt.hessian_approximation = 'limited-memory';  % 近似海森矩阵
    % options.ipopt.warm_start_init_point = 'yes';
    
    solver = nlpsol('solver', 'ipopt', nlp_prob, options);
    
    args   = struct;
    args.lbg(1:len_g,1)   =  zeros(size(g));  % Equality constraints, g, state_guess - state_calculated = 0
    args.ubg(1:len_g,1)   =  zeros(size(g));  % constraints, 
    args.lbx(1:x_num,1)   =  ones(x_num,1)*-inf; % lower and upper bound of states pt & S0t
    args.ubx(1:x_num,1)   =  ones(x_num,1)*+inf; 

    
    
    % ----------------------- Solver -------------------
    args.x0 = reshape(pk  ,x_num,1);
    
    % Initial values and Constraints
    solution = solver('x0', args.x0,...
                      'lbx',args.lbx, 'ubx',args.ubx,...
                      'lbg',args.lbg, 'ubg',args.ubg);
    % J           = full(solution.f);
    pk_opt          = reshape(full(solution.x(1:x_num)), x_num, []);

end

