function R3 = Quat2Rot(q, qref)
% 四元数 转 旋转矩阵
%QUAT2ROT 此处显示有关此函数的摘要
q1_inv = [qref(1), -qref(2), -qref(3), -qref(4)];
q_AB = quatmultiply(q1_inv, q);

q0 = q_AB(:,1);
q1 = q_AB(:,2);
q2 = q_AB(:,3);
q3 = q_AB(:,4);


%   此处显示详细说明
R3 = [
    1 - 2*(q2.^2 + q3.^2), 2*(q1.*q2 - q3.*q0),   2*(q1.*q3 + q2.*q0);
    2*(q1.*q2 + q3.*q0),   1 - 2*(q1.^2 + q3.^2), 2*(q2.*q3 - q1.*q0);
    2*(q1.*q3 - q2.*q0),   2*(q2.*q3 + q1.*q0),   1 - 2*(q1.^2 + q2.^2)
];

% R3 = [
%     1 - 2*(qy^2 + qz^2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw);
%     2*(qx*qy + qz*qw),  1 - 2*(qx^2 + qz^2),  2*(qy*qz - qx*qw);
%     2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx^2 + qy^2)
% ];


end


