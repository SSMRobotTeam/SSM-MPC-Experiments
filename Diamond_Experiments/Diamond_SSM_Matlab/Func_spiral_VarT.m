function zd = Func_spiral_VarT(t,k,a,b)
% % % % only outdofs will be nonzero
% 参数设置
    % a = 0.002;       % 控制螺旋的间距
    % % b = 2 * pi;    % 控制初始角速度
    % b = pi;    % 控制初始角速度
    % k = 0.5;       % 控制角速度的变化

    N = length(t);
    m = 3;
    zd = zeros(m,N); % x:1-y:2-z:3 motion of the end-effector
    r = a * t;              % 半径随时间线性增长
    theta = b * t.^k;       % 角度变化越到外圈越慢

    zd(1,:) = r .* cos(theta);    % x
    zd(2,:) = -r .* sin(theta);     % x
    zd(3,:) = 0; % z

end
