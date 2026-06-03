function Hx_ext = Hz_constraint(x, Node_list)

n      = length(x);
fixed_node  =  Node_list{end};
Hx     = zeros(4,n);

for i = 1:4
    Node_list{i}(1:3) = Node_list{i}(1:3) + x(Node_list{i}(4:6))';
    delta_xyz         = Node_list{i}(1:3) - fixed_node;
    Hx(i,Node_list{i}(4:6)) = delta_xyz / norm(delta_xyz,2);
end


Hx_ext = [Hx, zeros(size(Hx))];
end

