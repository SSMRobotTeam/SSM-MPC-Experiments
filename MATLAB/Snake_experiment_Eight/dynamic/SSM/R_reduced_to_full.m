function [z] = R_reduced_to_full(p,W0,W1,epsilon)

nt = size(p,2);

if isempty(W1) || epsilon == 0
    N = size(W0(1).coeffs ,1);
    z = zeros(N,1);
else
   
    % Nonautonomous first order time contributions
    num_kappa = numel(W1);
    phi = linspace(0,2*pi,nt); % assuming single periodic frequency
    
    N = size(W0{1}.coeffs ,1);
    z = zeros(N,1);
    
    for i = 1:num_kappa
        order = numel(W1(i).W);
        % zeroth order
        W10 = W1(i).W(1);
        z = z + epsilon * real( W10.coeffs * exp(1i * W1(i).kappa * phi));    % Higher order time contributions
        
        for j = 1:order-1 %array starting at 0
            Wij = W1(i).W(j+1);
            z = z + epsilon * real(expand_multiindex(Wij,p) .* exp(1i * W1(i).kappa * phi));
        end
        
    end 
end

for j = 1:length(W0)
    if strcmp(class(p),'casadi.SX')
        %  || max(abs(imag(p)))==0
        % if we use CasADi, we separate the real part and imaginary part 
        % to compute the SSM mapping W(p), so the input p, must contain the
        % real part 'p_real' and imaginary part 'p_imag'
        z =  z + (R_expand_multiindex(W0(j),p));
    else
        z =  z + real(R_expand_multiindex(W0(j),p)); 
    end
end

end
