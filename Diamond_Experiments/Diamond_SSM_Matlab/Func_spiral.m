function zd = Func_spiral(t,T,a,b)
% % % % only outdofs will be nonzero
% a: initial radius
% b: increasing factor
    if t <= (2*T)
        N = length(t);
        m = 3;
        zd = zeros(m,N); % x:1-y:2-z:3 motion of the end-effector
        theta = t/T*(2*pi);
        r     = a + b.*theta;
        zd(2,:) = r.*cos(theta);       % x
        zd(1,:) = r.*sin(theta);       % y
        zd(3,:) = 0; % z
        % zd(3,:) = -0.05 * r.*sin(1*theta); % z
    else
        r  = a + b.*4*pi; 
        % y0 = r.*cos(4*pi);       % x
        % x0 = r.*sin(4*pi);       % y
        % R  = norm([x0, y0]);        % R

        alpha = mod(t/(2*T),1)*(2*pi);
        zd(2,:) = r.*cos(alpha);  
        zd(1,:) = r.*sin(alpha);  
        zd(3,:) = 0;
    end
end
