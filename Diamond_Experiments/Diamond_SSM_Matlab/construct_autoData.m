function autData = construct_autoData(R_0)
% CONSTRUCT_AUTODATA This function constructs autoData that is used for
% reduced dynamics

% extract coefficients and exponents
beta  = [];
kappa = [];
for k = 2:numel(R_0)
    R = R_0(k);
    betak = R.coeffs;
    if ~isempty(betak)
        kappak = R.ind;
        % assemble terms
        beta  = [beta betak];
        kappa = [kappa; kappak];
    end
end
autData       = struct();
% autData.lamd  = DS.spectrum.Lambda(modes);
lam           = R_0(1).coeffs;
autData.lamd  = [lam(1,4); lam(2,3); lam(3,2); lam(4,1)];
autData.beta  = beta;
autData.kappa = kappa;

end
