function dist = CableLength_double_part(x,Node_List)
%CONSTRAINT_H 此处显示有关此函数的摘要
%   此处显示详细说明
% x: displacement of nodes
m_long  = 3;  n_long  = size(Node_List{1},2);
m_short = 3;  n_short = size(Node_List{2},2); 
% update the active node coordinates
for j = 1:m_long      % 绳子编号
    for i = 2:n_long  % 节点编号
        ind = (j-1)*48 + 3*(i-1);
        Node_List{1}{j,i}(1:3) = Node_List{1}{j,i}(1:3) + x(ind-2:ind);
    end
end

for j = 1:m_short
    for i = 2:n_short
        ind = (j-1)*27 + 3*(i-1) + 48*3;
        Node_List{2}{j,i}(1:3) = Node_List{2}{j,i}(1:3) + x(ind-2:ind);
    end
end

dist = zeros(m_long+m_short,1);
% calculate the length of the cables
for j = 1:m_long
    for i = 1:n_long-1
        dist(j) = dist(j) + norm(Node_List{1}{j,i}(1:3)-Node_List{1}{j,i+1}(1:3),2);
    end
end

for j = 1:m_short
    for i = 1:n_short-1
        dist(j+3) = dist(j+3) + norm(Node_List{2}{j,i}(1:3)-Node_List{2}{j,i+1}(1:3),2);
    end
end

end

