function zout = Function_TrajdTraj(t,T1,Teight,P_start,dP_start,P_end,dP_end,A,omega,name)
    C0 = P_start;
    C1 = P_start + T1*dP_start/3;
    C2 = P_end   - T1*dP_end  /3;
    C3 = P_end;
    Coeffs = [C0';C1';C2';C3'];

    Bu    = @(t) [(1-(t/T1)).^3;...
                  3*(t/T1).*(1-(t/T1)).^2;...
                  3*((t/T1).^2).*(1-(t/T1));...
                  (t/T1).^3];
    dBu   = @(t) [-3/T1*(1-(t/T1)).^2;...
                  3/T1*(1-t/T1).*(1-3*(t/T1));...
                  (3*t/T1^2).*(2-3*t/T1);...
                  (3/T1).*(t/T1).^2];
    zout = zeros(3,length(t));
    % 找到t中第一个小于T的下标
    first_index = find(t > T1, 1);
    if isempty(first_index)
        first_index = length(t);

    elseif first_index == 1
        first_index = 0;
    end
    if name == "Traj"
            % B样条段
            t_1seg = t(1:first_index);
            zout(:,1:first_index) = (Bu(t_1seg)' * Coeffs)';
            % 8字轨迹段（τ = t-T）
            t_2seg = t(first_index+1:end) - T1;
%             theta = omega * t_2seg;  % 8字轨迹相位角
            theta = (t_2seg - 0.25*Teight) / Teight * (2*pi);  % 8字轨迹相位角
            zout(:,first_index+1:end) = [P_end(1) + 0.35*A * cos(theta) ./ (1 + sin(theta).^2);...  % x方向运动（满足初始条件）
                                         P_end(2) + zeros(size(theta));...                       % y固定（xz平面） 
                                         P_end(3) + A * cos(theta) .* sin(theta) ./ (1 + sin(theta).^2)...% z方向运动（2倍频率形成8字）
                                         ];
    elseif name == "dTraj"
        % B样条速度
        t_1seg = t(1:first_index);
        zout(:,1:first_index) = (dBu(t_1seg)' * Coeffs)';
        % 8字轨迹速度
        t_2seg = t(first_index+1:end) - T1;
        theta = (t_2seg - 0.25*Teight) / Teight * (2*pi);
        zout(:,first_index+1:end) = [
                                    -0.35 * A * omega * sin(theta) .* (3 - sin(theta).^2) ./ (1 + sin(theta).^2).^2;  % dx/dt
                                    zeros(size(theta));  % dy/dt
                                    A * omega * (1 - 3 * sin(theta).^2) ./ (1 + sin(theta).^2).^2  % dz/dt
                                ];

    end
end