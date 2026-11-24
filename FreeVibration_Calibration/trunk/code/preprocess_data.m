% Preprocess the preserved experimental input using the original geometry.
% Outputs time and displ_endeff_save in the current workspace.
code_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(code_dir);
addpath(code_dir);
csvFilePath = fullfile(project_dir, 'data', 'free_vibration_input.csv');
FreeVib_Data1 = readtable(csvFilePath);

free_mat = table2array(FreeVib_Data1(:,1:16));
% 找到包含 NaN 的行
rowsWithNaN = any(isnan(free_mat), 2);
% 删除包含 NaN 的行
free_mat = free_mat(~rowsWithNaN, :);


phi = 15;
c   = 8;  
d   = 20;
h   = 20;
e   = 3; % 末端盘的底部厚度
h_basis = h - phi/2;

time  = free_mat(:,2);
basec_xyz      = free_mat(1,7:9);    
basec_xyz(:,2) = basec_xyz(:,2) - h_basis; % y方向 偏移到机械臂基座端

endeffc_xyz = free_mat(:,14:16);
endeffc_q   = free_mat(:,10:13);

base_q        = [free_mat(1,6), free_mat(1,3), free_mat(1,4), free_mat(1,5)];
R_B2G         = quat2dcm(base_q); % 全局坐标系转基座坐标系

delta_EB    = (endeffc_xyz(1,:) - basec_xyz)' % 末端执行器 与 基座坐标系的差值，在G系中表示
pos_endeff_rigid  = (R_B2G*delta_EB)' % 参考位置，相对基座的位置

endeff_q0   =  [endeffc_q(1,4), endeffc_q(1,1), endeffc_q(1,2), endeffc_q(1,3)];
R3          = Quat2Rot(endeff_q0, base_q);
ey          = [0; 0; c-e];  % 末端执行器在末端局部坐标系下的局部坐标
end_bias    = R3 * ey;

pos_endeff_ref  = pos_endeff_rigid  + end_bias' % 末端刚体 ---> 实际末端位置

N = size(free_mat,1);
displ_endeff_save = [];
for i = 1:N
    endeff_xyz = endeffc_xyz(i,:);
    endeff_q   = [endeffc_q(i,4), endeffc_q(i,1), endeffc_q(i,2), endeffc_q(i,3)];

    R3          = Quat2Rot(endeff_q, base_q);
    end_bias    = R3 * ey;
    
    delta_EB    = (endeff_xyz - basec_xyz)'; % 末端执行器 与 基座坐标系的差值，在G系中表示
    pos_endeff  = (R_B2G*delta_EB)' + end_bias';        % 转到基座坐标系下
    
    displ_endeff = pos_endeff - pos_endeff_ref;
    displ_endeff_save = [displ_endeff_save; displ_endeff];
end

