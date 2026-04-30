function zd = eightCurve(t,T,A,outdof_endeff)
% % % % only outdofs will be nonzero
    N = length(t);
    m = length(outdof_endeff);
    zd = zeros(m,N); % x:1-y:2-z:3 motion of the end-effector
    t = (t-1/4*T)/T*(2*pi);
    zd(1,:) = 0.35*A * cos(t) ./ (1 + sin(t).^2);       % x
    zd(2,:) = A * cos(t) .* sin(t) ./ (1 + sin(t).^2); % z
    % for i = 1:N
    %     zd(2,:) = A * cos(t) .* sin(t) ./ (1 + sin(t).^2); % z
    % end
end
