function plot_setting2()
    box on
    set(gcf, 'Position', [100, 100, 700, 800]);
    % 设置刻度字体属性
    set(gca, 'FontSize', 24);
    ylim([-12.5, 17.5])
end