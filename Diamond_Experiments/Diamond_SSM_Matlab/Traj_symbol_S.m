function zd = Traj_symbol_S(t,dt)
k = round(t./dt);
load("x_S.mat","x_S")
load("y_S.mat","y_S")

N = length(t);
zd = zeros(3,N); % x:1-y:2-z:3 motion of the end-effector
zd(1,:) = x_S(k);       % x
zd(2,:) = y_S(k);       % y
zd(3,:) = zeros(1,length(k));      % z

zd = zd./1000;
end

