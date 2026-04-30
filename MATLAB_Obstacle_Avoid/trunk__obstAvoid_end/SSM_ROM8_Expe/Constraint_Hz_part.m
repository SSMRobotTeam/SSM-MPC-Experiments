function Hz = Constraint_Hz_part(x,Node_List)
%CONSTRAINT_H 此处显示有关此函数的摘要
%   此处显示详细说明
% z: displacement of nodes, 1 order system
m_long  = 3;  n_long  = size(Node_List{1},2);
m_short = 3;  n_short = size(Node_List{2},2); 
n     = length(x);
Hz    = zeros(6,n);
% 长绳
for j = 1:m_long
    for i = 2:n_long
        % 第一个点为固定端的坐标
        ind = (j-1)*48 + 3*(i-1);
%         disp(ind)
        if i==2
            AP1_xyz = Node_List{1}{j,i}(1:3)   + x(ind-2:ind);
            AP2_xyz = Node_List{1}{j,i-1}(1:3);
            delta_xyz1   = AP1_xyz - AP2_xyz;

            Hz(j,ind-2:ind) = Hz(j,ind-2:ind)   +   delta_xyz1' ./ norm(delta_xyz1,2);
        else
            AP1_xyz = Node_List{1}{j,i}(1:3)   + x(ind-2:ind);
            AP2_xyz = Node_List{1}{j,i-1}(1:3) + x(ind-3-2:ind-3);
            delta_xyz   = AP1_xyz - AP2_xyz;
            
            Hz(j,ind-2:ind)   = Hz(j,ind-2:ind)   +  delta_xyz' ./ norm(delta_xyz,2);
            Hz(j,ind-3-2:ind-3) = Hz(j,ind-3-2:ind-3) + -delta_xyz' ./ norm(delta_xyz,2);
        end
    end
end

% 短绳
for j = 1:m_short
    for i = 2:n_short
        ind = (j-1)*27 + 3*(i-1) + 48*3;
%         disp(ind)
        % 第一个点为固定端的坐标
        if i==2
            AP1_xyz = Node_List{2}{j,i}(1:3)   + x(ind-2:ind);
            AP2_xyz = Node_List{2}{j,i-1}(1:3);
            delta_xyz1   = AP1_xyz - AP2_xyz;

            Hz(j+3,ind-2:ind) = Hz(j+3,ind-2:ind)   +   delta_xyz1' ./ norm(delta_xyz1,2);
        else
            AP1_xyz = Node_List{2}{j,i}(1:3)   + x(ind-2:ind);
            AP2_xyz = Node_List{2}{j,i-1}(1:3) + x(ind-3-2:ind-3);
            delta_xyz   = AP1_xyz - AP2_xyz;
            
            Hz(j+3,ind-2:ind)   = Hz(j+3,ind-2:ind)   +  delta_xyz' ./ norm(delta_xyz,2);
            Hz(j+3,ind-3-2:ind-3) = Hz(j+3,ind-3-2:ind-3) + -delta_xyz' ./ norm(delta_xyz,2);
        end
    end
end
end
