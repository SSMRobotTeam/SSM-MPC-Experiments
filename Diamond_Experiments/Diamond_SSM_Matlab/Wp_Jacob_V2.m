function W_Jacob = Wp_Jacob_V2(Wmap,p0,N,W_pk0)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    % Jacobian by numerical method
    n     = length(p0); 
    W_Jacob = zeros(N,n); 
    xk    = p0;
    delta = eps;
    for k = 1:n
        xPerturbed    = xk;
        xPerturbed(k) = xPerturbed(k) + delta;
        % W_perturbed = C_reduced_to_full(xPerturbed,Wmap,[],0);
        W_perturbed = R_reduced_to_full(xPerturbed,Wmap,[],0);
        W_Jacob(:,k) = (W_perturbed - W_pk0) / delta;
    end
end

