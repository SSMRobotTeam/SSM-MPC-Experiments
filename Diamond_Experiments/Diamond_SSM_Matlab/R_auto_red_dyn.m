function y = R_auto_red_dyn(z, data)
% auto_red_dyn This function presents vectorized implementation of vector field
% of reduced dynamics of autonomous SSMs, namely, \dot{x}=R(x). The x
% vector here is a mixture of both real and complex-conjugate pair
% variables.

assert(~isempty(data), 'Structure data is empty');
% extract data fields
lamd  = data.lamd;
beta  = data.beta;
kappa = data.kappa;

%%  If the state is in double type
% y = lamd.*z;
% % nonlinear part
% nTerms = size(kappa,1);
% for i=1:nTerms
%    coeffs = beta(:,i);
%    if max(abs(coeffs))>1e-8 
%        terms  = z_power_ka(z, kappa(i,:));  % power of each element of p
%        y = y+coeffs*terms;
%    end
% end
% 

n      = size(z,1);
m      = size(z,2);
p_real = zeros(n,m);
p_imag = zeros(n,m);
% P_real = zeros(n,m);
% P_imag = zeros(n,m);
p_real(1:2:end-1,:) = z(1:2:end-1,:);
p_real(2:2:end  ,:) = z(1:2:end-1,:);
p_imag(1:2:end-1,:) = z(2:2:end,:);
p_imag(2:2:end  ,:) = -z(2:2:end,:);
% z_real = z(1:2:end-1,:);
% z_imag = z(2:2:end,:);

lamd_real = real(lamd);
lamd_imag = imag(lamd);
y_real = p_real.*lamd_real - p_imag.*lamd_imag; % n*1
y_imag = p_real.*lamd_imag + p_imag.*lamd_real; % n*1
% nonlinear part
nTerms = size(kappa,1); %
for i=1:nTerms
   coeffs = beta(:,i);
   coeffs_real = real(coeffs);
   coeffs_imag = imag(coeffs);
   if max(abs(coeffs))>1e-8 
       [term_real, term_imag] = z_power_ka_RIsep(p_real,p_imag, kappa(i,:)); 
       y_real      = y_real + (coeffs_real * term_real - coeffs_imag * term_imag);
       y_imag      = y_imag + (coeffs_real * term_imag + coeffs_imag * term_real);
   end
end

yout = zeros(n,m);
yout(1:2:end-1,:) = y_real(1:2:end-1,:);
yout(2:2:end,:)   = y_imag(1:2:end-1,:);
y = yout;


end

function y = z_power_ka(z, ka)
% Z_POWER_KA This function computes complex monomilal z^ka, where
% z=[z1,...,zm] and ka=[ka1,...,kam] and z^ka=z1^ka1*...*zm^kam. Here we
% support vectorized version of z^ka. Specifically, z could be a m-by-n
% matrix, where each row corresponds the component sampled at different
% locations. However, we assume ka to be a vector
m = size(z,1);
% assert(m==numel(ka), 'The dimension of z and ka is not matched');
y = 1;
for i=1:m
    y = y.*z(i,:).^ka(i);
end

end

function [y_real, y_imag] = z_power_ka_RIsep(z_real,z_imag, ka)
    m = size(z_real,1);
    y_real = 1;
    y_imag = 0;
    for i = 1:m
        [a2,b2]          = imag_exp(z_real(i), z_imag(i), ka(i));
        [y_real, y_imag] = imag_multi2(y_real,y_imag,a2,b2);
    end
end

function [x,y] = imag_exp(a,b,i)
    % % calculate x+iy = (a+ib)^i
    % % i is the integer exponential
    % theta   =  i * atan2(b, a);
    % r       =  (sqrt(a*a + b*b))^i;
    % x       =  r * cos(theta);
    % y       =  r * sin(theta);
    if i == 0
        x = 1;
        y = 0;
    elseif i ==1
            x = a;
            y = b;
    elseif i ==2
            x = (a.^2-b.^2);
            y = 2*a.*b;
    elseif i==3
            x = (a.^3-3*a.*b.^2);
            y = 3*a.^2.*b - b.^3;
    elseif i==4
            % (a+ib)^4 = (a+ib)^3 * (a+ib)
            %          = (x1+iy1) * (a+ib)
            x1 = (a.^3-3*a.*b.^2);
            y1 = 3*a.^2.*b - b.^3;
            
            x  =  x1*a - y1*b;
            y  =  x1*b + y1*a;
    elseif i==5
            % (a+ib)^5 = (a+ib)^3 * (a+ib)^3
            %          = (x1+iy1) * (x2+iy2)
            x1 = (a.^3-3*a.*b.^2);
            y1 = 3*a.^2.*b - b.^3;
            
            x2 = (a.^2-b.^2);
            y2 = 2*a.*b;

            x  = x1*x2 - y1*y2;
            y  = x1*y2 + x2*y1;
    elseif i==6
            % (a+ib)^5 = (a+ib)^3 * (a+ib)^3
            x1 = (a.^3-3*a.*b.^2);
            y1 = 3*a.^2.*b - b.^3;
            
            x  = x1.*x1 - y1.*y1;
            y  = 2*x1.*y1;
    elseif i==7
            % (a+ib)^5 = (a+ib)^3 * (a+ib)^3 * (a+ib)
            x1 = (a.^3-3*a.*b.^2);
            y1 = 3*a.^2.*b - b.^3;
            
            x2  = x1.*x1 - y1.*y1;
            y2  = 2*x1.*y1;

            x  =  x2*a - y2*b;
            y  =  x2*b + y2*a;
    end
end

function [x,y] = imag_multi2(a1,b1,a2,b2)
    % calculate x+iy = (a1+ib1)*(a2+ib2)
    x = a1*a2 - b1*b2;
    y = a1*b2 + a2*b1;
end
