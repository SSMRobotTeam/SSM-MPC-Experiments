function plot_setting2()
    box on
    set(gcf, 'Position', [100, 100, 700, 800]);
    % 设置刻度字体属性
    set(gca, 'FontSize', 24);
    xlim([-17.5, 12.5])
    ylim([-17.5, 12.5])
    % axis equal
end