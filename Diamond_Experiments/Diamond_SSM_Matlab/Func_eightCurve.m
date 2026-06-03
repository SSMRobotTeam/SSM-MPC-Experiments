function zd = Func_eightCurve(t,T,A)
% % % % only outdofs will be nonzero
    N = length(t);
    zd = zeros(2,N); % x:1-y:2-z:3 motion of the end-effector
    t = (t-1/4*T)/T*(2*pi);
    zd(2,:) = -0.35*A * cos(t) ./ (1 + sin(t).^2);       % x
    zd(1,:) = A * cos(t) .* sin(t) ./ (1 + sin(t).^2); % y
    zd(3,:) = 0; % y
    % for i = 1:N
    %     zd(2,:) = A * cos(t) .* sin(t) ./ (1 + sin(t).^2); % z
    % end
end
