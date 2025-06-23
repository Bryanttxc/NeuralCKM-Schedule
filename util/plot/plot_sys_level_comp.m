% created by Bryanttxc, FUNLab

function plot_sys_level_comp(config)

% Include paths of necessary folders
root_path = config.root_path;
addpath(fullfile(root_path, 'result', 'data', 'system_level', 'schedule_src_sample'));

%% load data

Proposed_data_numIRS_6 = load(['4_3_enhanced_SE_opt_Time_numIRS_', num2str(6), '_angle_120.mat']);
Gurobi_SE_6 = Proposed_data_numIRS_6.Total_minSE(:,1);
H_SE_6 = Proposed_data_numIRS_6.Total_minSE(:,2);
random_SE_6 = Proposed_data_numIRS_6.Total_minSE(:,3);

% % benchmark 4-1
% Bench_4_1_sort_data_numIRS_6 = load(['bench_4_1_sort_SE_opt_Time_numIRS_', num2str(6), '_angle_120.mat']);
% bench_4_1_sort_SE_6 = Bench_4_1_sort_data_numIRS_6.Total_minSE(:,1);
% 
% Bench_4_1_rand_data_numIRS_6 = load(['bench_4_1_random_SE_opt_Time_numIRS_', num2str(6), '_angle_120.mat']);
% bench_4_1_rand_SE_6 = Bench_4_1_rand_data_numIRS_6.Total_minSE(:,1);
% 
% % benchmark 4-2
% Bench_4_2_greedy_data_numIRS_6 = load(['bench_4_2_greedy_LP_SE_opt_Time_numIRS_', num2str(6), '_angle_120.mat']);
% bench_4_2_greedy_SE_6 = Bench_4_2_greedy_data_numIRS_6.Total_minSE(:,1);
% 
% Bench_4_2_Hungarian_data_numIRS_6 = load(['bench_4_2_Hungarian_LP_SE_opt_Time_numIRS_', num2str(6), '_angle_120.mat']);
% bench_4_2_Hungarian_SE_6 = Bench_4_2_Hungarian_data_numIRS_6.Total_minSE(:,1);
% 
% % benchmark 4-3
% Bench_4_3_greedy_data_numIRS_6 = load(['bench_4_3_greedy_SE_opt_Time_numIRS_', num2str(6), '_angle_120.mat']);
% bench_4_3_greedy_SE_6 = Bench_4_3_greedy_data_numIRS_6.Total_minSE(:,1);
% 
% Bench_4_3_enhanced_data_numIRS_6 = load(['bench_4_3_enhanced_SE_opt_Time_numIRS_', num2str(6), '_angle_120.mat']);
% bench_4_3_enhanced_SE_6 = Bench_4_3_enhanced_data_numIRS_6.Total_minSE(:,1);

% Proposed_data_numIRS_4 = load(['4_3_enhanced_SE_opt_Time_numIRS_', num2str(4), '_angle_120.mat']);
% Gurobi_SE_4 = Proposed_data_numIRS_4.Total_minSE(:,1);
% H_SE_4 = Proposed_data_numIRS_4.Total_minSE(:,2);
% random_SE_4 = Proposed_data_numIRS_4.Total_minSE(:,3);

Proposed_data_numIRS_2 = load(['4_3_enhanced_SE_opt_Time_numIRS_', num2str(2), '_angle_120.mat']);
Gurobi_SE_2 = Proposed_data_numIRS_2.Total_minSE(:,1);
H_SE_2 = Proposed_data_numIRS_2.Total_minSE(:,2);
random_SE_2 = Proposed_data_numIRS_2.Total_minSE(:,3);


%% Plot

num_UE_vec = config.num_UE_vec;
% color = config.four_color_choice2;
color = config.three_color_comp;
linewidth = config.linewidth;
markersize = config.markerSize;

figure;
set(gcf, 'color', 'w');
hold on

plot(num_UE_vec, Gurobi_SE_6, 'color', color(1,:), 'LineStyle', '--', 'LineWidth', linewidth, 'Marker', '+', 'MarkerSize',markersize);
plot(num_UE_vec, H_SE_6, 'Color', color(1,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'diamond', 'MarkerSize',markersize);
plot(num_UE_vec, random_SE_6, 'Color', color(2,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize',markersize);

% % benchmark 4-1
% plot(num_UE_vec, bench_4_1_sort_SE_6, 'Color', color(3,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize',markersize);
% plot(num_UE_vec, bench_4_1_rand_SE_6, 'Color', color(4,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'x', 'MarkerSize',markersize);

% % benchmark 4-2
% plot(num_UE_vec, bench_4_2_greedy_SE_6, 'Color', color(3,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize',markersize);
% plot(num_UE_vec, bench_4_2_Hungarian_SE_6, 'Color', color(4,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'x', 'MarkerSize',markersize);

% % benchmark 4-3
% plot(num_UE_vec, bench_4_3_greedy_SE_6, 'Color', color(3,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize',markersize);
% plot(num_UE_vec, bench_4_3_enhanced_SE_6, 'Color', color(4,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'x', 'MarkerSize',markersize);


% plot(UE_num, Gurobi_SE_4, 'Color', color(2,:), 'LineStyle', '--', 'LineWidth', linewidth, 'Marker', 'x', 'MarkerSize',markersize);
% plot(UE_num, H_SE_4, 'Color', color(2,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize',markersize);

plot(num_UE_vec, Gurobi_SE_2, 'Color', color(3,:), 'LineStyle', '--', 'LineWidth', linewidth, 'Marker', '*', 'MarkerSize',markersize);
plot(num_UE_vec, H_SE_2, 'Color', color(3,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'square', 'MarkerSize',markersize);
plot(num_UE_vec, random_SE_2, 'Color', color(2,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'x', 'MarkerSize',markersize);

xlim([30 210]);
% ylim([0.1 1.5]);
xlabel("Number of UEs", 'FontName', 'Times New Roman');
ylabel("Minimum Ergodic Throughput(bps/Hz)", 'FontName', 'Times New Roman');

% legend("numIRS-6-Gurobi", "numIRS-6-Proposed", "numIRS-6-Bench-4-1-sort", "numIRS-6-Bench-4-1-random", 'fontname', 'Times New Roman', 'fontsize', 35);
% legend("numIRS-6-Gurobi", "numIRS-6-Proposed", "numIRS-6-Bench-4-2-greedy-LP", "numIRS-6-Bench-4-2-Hungarian-LP", 'fontname', 'Times New Roman', 'fontsize', 35);
% legend("numIRS-6-Gurobi", "numIRS-6-Proposed", "numIRS-6-Bench-4-3-greedy", "numIRS-6-Bench-4-3-enhanced", 'fontname', 'Times New Roman', 'fontsize', 35);

legend("6-IRS case, Gurobi", "6-IRS case, Proposed", "6-IRS case, Random",...
    "2-IRS case, Gurobi", "2-IRS case, Proposed", "2-IRS case, Random", 'fontname', 'Times New Roman', 'fontsize', 35);
set(gca, 'fontsize', 30);

% % ZOOM
% axes('Position',[0.5 0.37 0.25 0.25]);
% box on;
% hold on;
% 
% h1 = plot(num_UE_vec, Gurobi_SE_6, 'color', color(1,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', '+', 'MarkerSize',markersize);
% h2 =plot(num_UE_vec, H_SE_6, 'Color', color(2,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'diamond', 'MarkerSize',markersize);
% h3 =plot(num_UE_vec, bench_4_3_greedy_SE_6, 'Color', color(3,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize',markersize);
% h4 =plot(num_UE_vec, bench_4_3_enhanced_SE_6, 'Color', color(4,:), 'LineStyle', '-', 'LineWidth', linewidth, 'Marker', 'x', 'MarkerSize',markersize);
% 
% xlim([90, 130]);
% ylim([0.2, 0.45]);
% title('')
% xlabel('')
% ylabel('')
% set(gca,'fontsize',24)


end
