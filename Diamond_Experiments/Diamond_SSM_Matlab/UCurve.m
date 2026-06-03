function zd= UCurve(t,T)
N  = length(t);
zd = zeros(3,N); % x:1-y:2-z:3 motion of the end-effector

% 新增：绘制字母U形状的轨迹（关键修改）
x_u = t/T*4-2;
y_u = - 1*x_u.^2 + 1;        % 二次函数生成U形轨迹（开口向上）

zd(1,:) = -(x_u+2)*27.5./1000; % max:120,120
zd(2,:) = (y_u+3)*27.5./1000;
zd(3,:) = zeros(N,1); % y

end

