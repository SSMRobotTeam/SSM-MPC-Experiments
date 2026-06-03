function Hz = Control_Mat_H_casadi(x,Node_List)
%CONSTRAINT_H 此处显示有关此函数的摘要
import casadi.*
%   此处显示详细说明
% x: displacement of nodes
m     = size(Node_List,1);   % cable number
N     = size(Node_List,2)-1; % nodes number, last one is cable length
n     = length(x);
Hx    = SX.zeros(m,n);

% for j = 1:m
%     for k = 2:N
%         Node_List{j,k}(1:3) = Node_List{j,k}(1:3) + x(Node_List{j,k}(4:6))';
%     end
% end


for j = 1:m
    for i = 2:N
        if i==N
            % N-1<--N|
            AP2_xyz = Node_List{j,N}(1:3)    + x(Node_List{j,N}(4:6))';
            AP1_xyz = Node_List{j,N-1}(1:3)  + x(Node_List{j,N-1}(4:6))';

            delta_xyz12   = AP1_xyz - AP2_xyz;
            
            Hx(j,Node_List{j,N}(4:6)) = delta_xyz12 ./ norm(delta_xyz12,2);
        elseif i==2
            AP2_xyz = Node_List{j,i}(1:3)    + x(Node_List{j,i}(4:6))';
            AP1_xyz = Node_List{j,i-1}(1:3);
            AP3_xyz = Node_List{j,i+1}(1:3)  + x(Node_List{j,i+1}(4:6))';

            delta_xyz12   = AP1_xyz - AP2_xyz;
            delta_xyz32   = AP3_xyz - AP2_xyz;
            
            Hx(j,Node_List{j,i}(4:6))   = delta_xyz12 ./ norm(delta_xyz12,2)   +   delta_xyz32 ./ norm(delta_xyz32,2);
        else
            % 1<--2o2-->3
            AP2_xyz = Node_List{j,i}(1:3)    + x(Node_List{j,i}(4:6))';
            AP1_xyz = Node_List{j,i-1}(1:3)  + x(Node_List{j,i-1}(4:6))';
            AP3_xyz = Node_List{j,i+1}(1:3)  + x(Node_List{j,i+1}(4:6))';

            delta_xyz12   = AP1_xyz - AP2_xyz;
            delta_xyz32   = AP3_xyz - AP2_xyz;
            
            Hx(j,Node_List{j,i}(4:6))   = delta_xyz12 ./ norm(delta_xyz12,2)   +   delta_xyz32 ./ norm(delta_xyz32,2);
        end
    end
end
Hz = [Hx'; zeros(size(Hx'))];
end
