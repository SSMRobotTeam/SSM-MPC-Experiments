function S = R_expand_multiindex(M,P)
%  EXPAND_MULTIINDEX This function returns the multi-index expansion $\mathbf{S}$ for the multi-index
% pair $(\mathbf{C},\mathbf{I})$ over the column vectors in the matrix $\mathbf{P}=[\mathbf{p}_1,\dots,\mathbf{p}_{n_t}]\in
% \mathbb{R}^{n\times n_t}$, where $\mathbf{C},\mathbf{I}$ are fields of
% the data structure M.
%
% $\mathbf{s}_j = \mathbf{C} (\mathbf{p}_j)^{\mathbf{I}},\quad j=1,\dots,n_t$,
%
% $$\mathbf{S}=[\mathbf{s}_1,\dots,\mathbf{s}_{n_t}]$$

import casadi.*

if isempty(M.ind)
    S = 0;
else
    % nonl_num = round(size(P,1) /2); % number of nonlinear modes, real + imag 

    % evaluate monomials
    % if strcmp(class(P), 'casadi.MX')
%% If the state is CasADi type
    % if isa(class(P,'casadi.SX')
    if isa(P,'casadi.SX')
        nt = size(P,2);
        C = M.coeffs;
        mt= size(C,2);
        I = M.ind;
        n_I = size(I,1);

        n      = size(P,1);
        m      = size(P,2);
        P_real = SX.zeros(n,m);
        P_imag = SX.zeros(n,m);
        % P_real = zeros(n,m);
        % P_imag = zeros(n,m);
        P_real(1:2:end-1,:) = P(1:2:end-1,:);
        P_real(2:2:end  ,:) = P(1:2:end-1,:);
        P_imag(1:2:end-1,:) = P(2:2:end,:);
        P_imag(2:2:end  ,:) = -P(2:2:end,:);

        % [P_real,P_imag] = ztransformation(P);
        % P_real = P(1 : m_I);
        % P_imag = P(m_I + 1 : 2*m_I);
        P_real_kron = kron(P_real.',ones(n_I,1));
        P_imag_kron = kron(P_imag.',ones(n_I,1));
        index_multi = repmat(I,nt,1);

        [row,col] = size(P_real_kron);
        P_real_multi = SX.zeros(row,col);
        P_imag_multi = SX.zeros(row,col);
        % P_real_multi = zeros(row,col);
        % P_imag_multi = zeros(row,col);
        
        for i = 1:row
            for j = 1:col
                a     = P_real_kron(i,j);
                b     = P_imag_kron(i,j);
                x     = index_multi(i,j);
                [P_real_multi(i,j), P_imag_multi(i,j)] = imag_exp(a,b,x);
            end
        end
        % product of each row
        % P_I_real   = zeros(row,1);
        % P_I_imag   = zeros(row,1);
        P_I_real   = SX.zeros(row,1);
        P_I_imag   = SX.zeros(row,1);
        for i = 1:row
            % In Diamond robot case, we choose 3 configuration, i.e. 6 dofs
            if col == 6
                a1 = P_real_multi(i,1); b1 = P_imag_multi(i,1);
                a2 = P_real_multi(i,2); b2 = P_imag_multi(i,2);
                a3 = P_real_multi(i,3); b3 = P_imag_multi(i,3);
                [x1,y1] = imag_3multi(a1,b1,a2,b2,a3,b3);
                
                a4 = P_real_multi(i,4); b4 = P_imag_multi(i,4);
                a5 = P_real_multi(i,5); b5 = P_imag_multi(i,5);
                a6 = P_real_multi(i,6); b6 = P_imag_multi(i,6);
                [x2,y2] = imag_3multi(a4,b4,a5,b5,a6,b6);

                P_I_real(i) = x1*x2 - y1*y2;
                P_I_imag(i) = x1*y2 + x2*y1;
            elseif col == 2
                x1 = P_real_multi(i,1); y1 = P_imag_multi(i,1);
                x2 = P_real_multi(i,2); y2 = P_imag_multi(i,2);

                P_I_real(i) = x1*x2 - y1*y2;
                P_I_imag(i) = x1*y2 + x2*y1;
            elseif col == 4
                a1 = P_real_multi(i,1); b1 = P_imag_multi(i,1);
                a2 = P_real_multi(i,2); b2 = P_imag_multi(i,2);
                a3 = P_real_multi(i,3); b3 = P_imag_multi(i,3);
                a4 = P_real_multi(i,4); b4 = P_imag_multi(i,4);
                [P_I_real(i),P_I_imag(i)] = imag_4multi(a1,b1,a2,b2,a3,b3,a4,b4);

            elseif col == 8
                a1 = P_real_multi(i,1); b1 = P_imag_multi(i,1);
                a2 = P_real_multi(i,2); b2 = P_imag_multi(i,2);
                a3 = P_real_multi(i,3); b3 = P_imag_multi(i,3);
                a4 = P_real_multi(i,4); b4 = P_imag_multi(i,4);
                [P1real,P1imag] = imag_4multi(a1,b1,a2,b2,a3,b3,a4,b4);

                a5 = P_real_multi(i,5); b5 = P_imag_multi(i,5);
                a6 = P_real_multi(i,6); b6 = P_imag_multi(i,6);
                a7 = P_real_multi(i,7); b7 = P_imag_multi(i,7);
                a8 = P_real_multi(i,8); b8 = P_imag_multi(i,8);
                [P2real,P2imag] = imag_4multi(a5,b5,a6,b6,a7,b7,a8,b8);
                
                P_I_real(i) = P1real * P2real - P1imag * P2imag;
                P_I_imag(i) = P1real * P2imag + P2real * P1imag;
            end

                
        end
        C_real = real(C);
        C_imag = imag(C);
        P_I_real = reshape(P_I_real,[mt,nt]);
        P_I_imag = reshape(P_I_imag,[mt,nt]);
        % S = C*reshape(P_I,[mt,nt]);
        S   = C_real*P_I_real - C_imag*P_I_imag;
    else
%% If the state is double type
        nt = size(P,2);
        C = M.coeffs;
        mt= size(C,2);
        I = M.ind;
        n_I = size(I,1);

        n      = size(P,1);
        m      = size(P,2);
        P_real = zeros(n,m);
        P_imag = zeros(n,m);
        % P_real = zeros(n,m);
        % P_imag = zeros(n,m);
        P_real(1:2:end-1,:) = P(1:2:end-1,:);
        P_real(2:2:end  ,:) = P(1:2:end-1,:);
        P_imag(1:2:end-1,:) = P(2:2:end,:);
        P_imag(2:2:end  ,:) = -P(2:2:end,:);

        % [P_real,P_imag] = ztransformation(P);
        % P_real = P(1 : m_I);
        % P_imag = P(m_I + 1 : 2*m_I);
        P_real_kron = kron(P_real.',ones(n_I,1));
        P_imag_kron = kron(P_imag.',ones(n_I,1));
        index_multi = repmat(I,nt,1);

        [row,col] = size(P_real_kron);
        P_real_multi = zeros(row,col);
        P_imag_multi = zeros(row,col);
        % P_real_multi = zeros(row,col);
        % P_imag_multi = zeros(row,col);
        
        for i = 1:row
            for j = 1:col
                a     = P_real_kron(i,j);
                b     = P_imag_kron(i,j);
                x     = index_multi(i,j);
                [P_real_multi(i,j), P_imag_multi(i,j)] = imag_exp(a,b,x);
            end
        end
        % product of each row
        % P_I_real   = zeros(row,1);
        % P_I_imag   = zeros(row,1);
        P_I_real   = zeros(row,1);
        P_I_imag   = zeros(row,1);
        for i = 1:row
            % In Diamond robot case, we choose 3 configuration, i.e. 6 dofs
            if col == 6
                a1 = P_real_multi(i,1); b1 = P_imag_multi(i,1);
                a2 = P_real_multi(i,2); b2 = P_imag_multi(i,2);
                a3 = P_real_multi(i,3); b3 = P_imag_multi(i,3);
                [x1,y1] = imag_3multi(a1,b1,a2,b2,a3,b3);
                
                a4 = P_real_multi(i,4); b4 = P_imag_multi(i,4);
                a5 = P_real_multi(i,5); b5 = P_imag_multi(i,5);
                a6 = P_real_multi(i,6); b6 = P_imag_multi(i,6);
                [x2,y2] = imag_3multi(a4,b4,a5,b5,a6,b6);

                P_I_real(i) = x1*x2 - y1*y2;
                P_I_imag(i) = x1*y2 + x2*y1;
            elseif col == 2
                x1 = P_real_multi(i,1); y1 = P_imag_multi(i,1);
                x2 = P_real_multi(i,2); y2 = P_imag_multi(i,2);

                P_I_real(i) = x1*x2 - y1*y2;
                P_I_imag(i) = x1*y2 + x2*y1;
            elseif col == 4
                a1 = P_real_multi(i,1); b1 = P_imag_multi(i,1);
                a2 = P_real_multi(i,2); b2 = P_imag_multi(i,2);
                a3 = P_real_multi(i,3); b3 = P_imag_multi(i,3);
                a4 = P_real_multi(i,4); b4 = P_imag_multi(i,4);
                [P_I_real(i),P_I_imag(i)] = imag_4multi(a1,b1,a2,b2,a3,b3,a4,b4);

            elseif col == 8
                a1 = P_real_multi(i,1); b1 = P_imag_multi(i,1);
                a2 = P_real_multi(i,2); b2 = P_imag_multi(i,2);
                a3 = P_real_multi(i,3); b3 = P_imag_multi(i,3);
                a4 = P_real_multi(i,4); b4 = P_imag_multi(i,4);
                [P1real,P1imag] = imag_4multi(a1,b1,a2,b2,a3,b3,a4,b4);

                a5 = P_real_multi(i,5); b5 = P_imag_multi(i,5);
                a6 = P_real_multi(i,6); b6 = P_imag_multi(i,6);
                a7 = P_real_multi(i,7); b7 = P_imag_multi(i,7);
                a8 = P_real_multi(i,8); b8 = P_imag_multi(i,8);
                [P2real,P2imag] = imag_4multi(a5,b5,a6,b6,a7,b7,a8,b8);
                
                P_I_real(i) = P1real * P2real - P1imag * P2imag;
                P_I_imag(i) = P1real * P2imag + P2real * P1imag;
            end

                
        end
        C_real = real(C);
        C_imag = imag(C);
        P_I_real = reshape(P_I_real,[mt,nt]);
        P_I_imag = reshape(P_I_imag,[mt,nt]);
        % S = C*reshape(P_I,[mt,nt]);
        S   = C_real*P_I_real - C_imag*P_I_imag;


        % nt = size(P,2);
        % C = M.coeffs;
        % I = M.ind;
        % n_I = size(I,1);
        % 
        % P_I = prod(kron(P.',ones(n_I,1)) .^ repmat(I,nt,1),2); % dim = 2 : MULTIPLY by row 
        % % evaluate polynomials
        % S = C*reshape(P_I,[],nt);
    end

    % 

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
            % (a+ib)^5 = (a+ib)^3 * (a+ib)^2
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
    elseif i<0
            x = 0;
            y = 0;
    end
end

function [x,y] = imag_4multi(a1,b1,a2,b2,a3,b3,a4,b4)
    % x + iy = (a1+ib1)(a2+ib2)(a3+ib3)(a4+ib4)
    % only for 4 complex multiply
    x  =  (a1*a2-b1*b2)*(a3*a4-b3*b4) - (a1*b2+b1*a2)*(a3*b4+b3*a4);
    y  =  (a1*a2-b1*b2)*(a3*b4+b3*a4) + (a1*b2+b1*a2)*(a3*a4-b3*b4);
end

function [x,y] = imag_3multi(a1,b1,a2,b2,a3,b3)
    % x + iy = (a1+ib1)(a2+ib2)(a3+ib3)(a4+ib4)
    % only for 4 complex multiply
    x  =  (a1*a2-b1*b2)*a3 - (a1*b2+a2*b1)*b3;
    y  =  (a1*a2-b1*b2)*b3 + (a1*b2+a2*b1)*a3;
end


function [z_real, z_imag] = ztransformation(z)
    n      = size(z,1);
    m      = size(z,2);
    z_real = DM.zeros(n,m);
    z_imag = DM.zeros(n,m);

    z_real(1:2:end-1,:) = z(1:2:end-1,:);
    z_real(2:2:end  ,:) = z(1:2:end-1,:);
    z_imag(1:2:end-1,:) = z(2:2:end,:);
    z_imag(2:2:end  ,:) = -z(2:2:end,:);

    % z_real = [z(1:2:end)';...
    %           z(1:2:end)'];
    % z_real = reshape(z_real,1,n)';
    % z_imag = [z(2:2:end)';...
    %           -z(2:2:end)'];
    % z_imag = reshape(z_imag,1,n)';
    % ztrans = [z_real;...
    %           z_imag];
end