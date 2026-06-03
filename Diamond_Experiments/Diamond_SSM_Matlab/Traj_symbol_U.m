function zd = Traj_symbol_U(t,dt)
k = round(t./dt);
load("x_U.mat","x_U")
load("y_U.mat","y_U")

N = length(t);
zd = zeros(3,N); % x:1-y:2-z:3 motion of the end-effector
zd(1,:) = x_U(k);       % x
zd(2,:) = y_U(k);       % y
zd(3,:) = zeros(1,length(k));      % z

zd = zd./1000;
end

