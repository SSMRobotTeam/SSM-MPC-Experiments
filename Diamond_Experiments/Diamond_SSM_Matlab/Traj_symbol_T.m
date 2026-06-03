function zd = Traj_symbol_T(t,dt)
k = round(t./dt);
load("x_T.mat","x_T")
load("y_T.mat","y_T")

N = length(t);
zd = zeros(3,N); % x:1-y:2-z:3 motion of the end-effector
zd(1,:) = x_T(k);       % x
zd(2,:) = y_T(k);       % y
zd(3,:) = zeros(1,length(k));      % z

zd = zd./1000;
end

