function dist = Cable_length(x, Node_list)
%CABLE_LENGTH 此处显示有关此函数的摘要
%   此处显示详细说明
fixed_node  =  Node_list{end};
dist   = zeros(4,1);
for i = 1:4
    Node_list{i}(1:3) = Node_list{i}(1:3) + x(Node_list{i}(4:6))';
    delta_xyz         = fixed_node - Node_list{i}(1:3);
    dist(i)           = norm(delta_xyz,2);
end

end

