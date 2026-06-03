function zd = SCurve(t,T)
%SCURVE 此处显示有关此函数的摘要
N = length(t);
zd = zeros(2,N); % x:1-y:2-z:3 motion of the end-effector

y_s = t/T*4-2;
x_s = 0.5*y_s.^3 - y_s;      % 三次多项式生成S形曲线（x关于y的函数）

zd(1,:) = -(x_s*24+48)./1000; % max:120,120
zd(2,:) =  (y_s*24+48)./1000;
zd(3,:) = zeros(N,1); % y

end

