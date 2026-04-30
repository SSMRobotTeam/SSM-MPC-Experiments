function Dynamic_capture_End_Func
%DYNAMIC_CAPTURE_START_FUNC 此处显示有关此函数的摘要
%   此处显示详细说明
%   初始化 posi
global y_exprm
plot_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'PlotFig');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
% save("t_exprm.mat","t_exprm")
save(fullfile(plot_dir, "y_exprm.mat"), "y_exprm")

end

