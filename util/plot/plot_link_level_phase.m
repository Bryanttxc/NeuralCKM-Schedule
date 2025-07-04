% created by Bryanttxc, FUNLab

function plot_link_level_phase(config)
%PLOT_LINK_LEVEL_PHASE plot the phase comparison

% Include paths of necessary folders
root_path = config.root_path;
addpath(fullfile(root_path, 'result', 'data', 'link_level', 'phaseComp'));

% % NN-1 sample
py_interpreter = config.py_interpreter;
py_uesenet_main_path = config.py_uesenet_main_path;
UE_vec = 1;
num_lab = 1;
max_num_sublab = 1;
[~, NN_1_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
                        ' --model ', 'vit', ...
                        ' --task ', 'cal', ...
                        ' --model_name ', 'vit.pth', ...
                        ' --UE_list ', [num2str(UE_vec)], ...
                        ' --num_lab ', num2str(num_lab), ...
                        ' --max_num_sublab ', num2str(max_num_sublab)])

suffix = '';

% elem = 8:2:16;
% cdf_data = [];
% for p = 1:length(elem)
%     path = ['link_level_CDF_sc_1333_from_AIRS_', num2str(elem(p)), 'x', num2str(elem(p)), '_BW_20MHz_SC_15KHz_fc_3p5GHz_approx', suffix, '.mat'];
%     result_cdf = load(path).result_cdf;
%     cdf_data = [cdf_data; result_cdf];
% end
% save("net/UESENet/result/data/cal/output/cdf_data_lab_1_1_numUE_1.mat", "cdf_data");

% NN-2 sample
config = gen_link_level_phase_sample(config, 2);
[~, NN_2_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
                        ' --model ', 'cdf', ...
                        ' --task ', 'cal', ...
                        ' --model_name ', 'cdf.pth', ...
                        ' --UE_list ', [num2str(UE_vec)], ...
                        ' --num_lab ', num2str(num_lab), ...
                        ' --max_num_sublab ', num2str(max_num_sublab)])

SE_data = load("net\UESENet\result\data\cal\output\SE_data_lab_1_1_numUE_1.mat").SE_data;
CKM_output = SE_data(:,end)

% load data
elem = 8:2:16;
num_case = length(elem);
num_scheme = 4;
elem_output = zeros(num_case, num_scheme);
for p = 1:num_case
    tmp = table2array(load(['link_level_SE_sc_1333_from_AIRS_', num2str(elem(p)), 'x', num2str(elem(p)), ...
                            '_BW_20MHz_SC_15KHz_fc_3p5GHz_approx', suffix, '.mat']).ergodic_SE_MC_Table);
    elem_output(p,:) = mean(tmp, 1);
end
% elem_output(num_case,1:2) = elem_output(num_case,1:2)-0.3;

% plot
figure;
set(gcf, 'color', 'w');

elem_vec = 8:2:16;
fig_x = elem_vec.^2;

markersize = config.markersize;
linewidth = config.linewidth;
fontsize = config.fontsize;
color = config.fivecolor;

plot(fig_x, elem_output(:,1)', 'color', color(1,:), 'Marker','+', 'MarkerSize', markersize, 'LineStyle','-', 'LineWidth', linewidth);
hold on
plot(fig_x, CKM_output', 'color', color(2,:), 'Marker','o', 'MarkerSize', markersize, 'LineStyle','--', 'LineWidth', linewidth);
plot(fig_x, elem_output(:,2)', 'color', color(3,:), 'Marker','*', 'MarkerSize', markersize, 'LineStyle',':', 'LineWidth', linewidth);
plot(fig_x, elem_output(:,3)', 'color', color(4,:), 'Marker','square', 'MarkerSize', markersize, 'LineStyle', '-.', 'LineWidth', linewidth);
plot(fig_x, elem_output(:,4)', 'color', color(5,:), 'Marker','x', 'MarkerSize', markersize, 'LineStyle','-', 'LineWidth', linewidth); % loss
xlabel("Number of AIRS Elements", 'FontName', 'Times New Roman')
ylabel("Ergodic Throughput(bps/Hz)", 'FontName', 'Times New Roman');
legend("Approx Formula Eq.(18)", "Neural CKM", "MCCM-based", "LoS beamforming", "Random phase", "Location", "best", 'FontName', 'Times New Roman');
set(gca, 'fontsize', fontsize);

end
