function zd = Func_spiral(t,T,a,b)
% % % % only outdofs will be nonzero
% a: initial radius
% b: increasing factor
    N = length(t);
    m = 3;
    zd = zeros(m,N); % x:1-y:2-z:3 motion of the end-effector
    theta = t/T*(2*pi);
    r     = a + b.*theta;
    zd(2,:) = r.*cos(theta);       % x
    zd(1,:) = r.*sin(theta);       % y
    zd(3,:) = 0; % z
    % zd(3,:) = -0.05 * r.*sin(1*theta); % z
end
