function R_Jacob = Rp_Jacob_V2(audata,p0,R_pk0)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    % Jacobian by numerical method
    n     = length(p0); 
    R_Jacob = zeros(n,n); 
    xk    = p0;
    delta = eps;
    for k = 1:n
        xPerturbed    = xk;
        xPerturbed(k) = xPerturbed(k) + delta;
        % R_perturbed   = C_auto_red_dyn(xPerturbed,audata);
        R_perturbed   = R_auto_red_dyn(xPerturbed,audata);
        R_Jacob(:,k)  = (R_perturbed - R_pk0) / delta;
    end
end

