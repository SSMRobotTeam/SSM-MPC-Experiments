function zd = Spiral3D(radius_max,height,omega,totalTime,t)
%SPIRAL3D 此处显示有关此函数的摘要
%   此处显示详细说明
% 螺旋轨迹参数
    % x = radius * cos(omega * t);
    % y = radius * sin(omega * t);
    % z = (height / totalTime) * t;  % 线性增加的高度
    % 
    % % 创建图形
    % posi_out = [x;y;z];
    radius = radius_max * (t / totalTime);
    N = length(t);
    zd = zeros(3,N); % x:1-y:2-z:3 motion of the end-effector
    zd(1,:) = 1e-3*radius .* cos(omega * t);       % x
    zd(2,:) = 1e-3*radius .* sin(omega * t);       % y
    zd(3,:) = 1e-3*((height / totalTime) * t + 7.5);      % z

end

