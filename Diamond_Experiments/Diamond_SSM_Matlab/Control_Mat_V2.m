function [Bext, Node_List_new] = Control_Mat_V2(x,Node_List)
%CONTROL_MAT 
% x is the difference of the displacement to the last simulation
% Node_list
% d1-d3:  centriod point as active node
% d4:d15: coords of 4 nodes of the closest tetraheron
% d16-d27:dofs of 4 nodes of the closest tetraheron (under constrained nodes number)
N = length(Node_List); % 18 anchor nodes
m = size(Node_List,1);
%% Update the Node_list according to the displacement field
for j = 1:m
    for k = 2:N
        % update the active node coordinates
        Node_List{j,k}(1:3) = Node_List{j,k}(1:3) + x(Node_List{j,k}(4:6))';
    end
end
Node_List_new = Node_List;
%% calculate the control matrix
n = length(x);
Bext = zeros(n,3); % 3 cables
for j = 1:m
    for i = 2:N
        % calculate the sumation of the cable force
        anchor = i;
        if i ~= N
            % anchor nodes
            P1 = Node_List{j,anchor-1}(1:3);
            P2 = Node_List{j,anchor}(1:3);
            P3 = Node_List{j,anchor+1}(1:3);
            % The direction of the cable force at section i, Regularization of the force direction
            R  = Regl_Vec(P1 - P2) + Regl_Vec(P3 - P2); 

            Bext(Node_List{j,anchor}(4:6),j) = R;
        else
            P1 = Node_List{j,anchor-1}(1:3);
            P2 = Node_List{j,anchor}(1:3);
            R  = Regl_Vec(P1 - P2);
            Bext(Node_List{j,anchor}(4:6),j) = R;
        end
    end
end
end


function y = Regl_Vec(x)
    y = x / norm(x,2);
end


% for j = 1:m
%     for i = 1:round(N/2)
%         % calculate the sumation of the cable force
%         if i ~= round(N/2)
%             anchor_1 = round(2*i);
%             anchor_2 = anchor_1 + 1;
%             % anchor nodes
%             P1 = Node_List{j,anchor_1-1}(1:3);
%             P2 = Node_List{j,anchor_1}(1:3);
%             P3 = Node_List{j,anchor_1+1}(1:3);
%             P4 = Node_List{j,anchor_1+2}(1:3);
%             % The direction of the cable force at section i, Regularization of the force direction
%             R  = Regl_Vec(P4 - P3) + Regl_Vec(P1 - P2); 
% 
%             %%% Plot the cables
%             % figure; hold on
%             % plot3([P1(1), P2(1)],[P1(2), P2(2)],[P1(3), P2(3)], 'b')
%             % plot3([P3(1), P4(1)],[P3(2), P4(2)],[P3(3), P4(3)], 'b')
%             % xlabel('x')
%             % ylabel('y')
%             % zlabel('z')
%             % % axis equal
%             % axis tight
% 
%             Bext(Node_List{j,anchor_1}(4:6),j) = 0.5*R;
%             Bext(Node_List{j,anchor_2}(4:6),j) = 0.5*R;
% 
%         else
%             anchor   = round(2*i);
%             P1 = Node_List{j,anchor-1}(1:3);
%             P2 = Node_List{j,anchor}(1:3);
%             R  = Regl_Vec(P1 - P2);
% 
%             Bext(Node_List{j,anchor}(4:6),j) = R;
%         end
%     end
% end