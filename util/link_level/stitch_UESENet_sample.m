% created by Bryanttxc, FUNLab

function stitch_UESENet_sample(config)
% STITCH_UESENET_SAMPLE stitch UESENet sample

%% load data

src_path_1 = fullfile(config.link_train_path, 'withBS_17feat_250117_viT_Sample_link_plus_system_level_1.mat');
src_path_2 = fullfile(config.link_train_path, 'withBS_17feat_back_250117_viT_Sample_link_plus_system_level.mat');
src_path_3 = fullfile(config.link_train_path, 'withBS_17feat_mid_250117_viT_Sample_link_plus_system_level.mat');
src_data_path = {src_path_1; src_path_2; src_path_3};

num_src = length(src_data_path);

%% NN-1

LPS_Net_sample = [];
LPS_Net_id = 1;
for src_idx = 1:num_src
    UE_cell = load(src_data_path{src_idx}).ue_cell;
    select_sample = UE_cell(:, LPS_Net_id);
    non_empty = ~cellfun(@isempty, select_sample);
    LPS_Net_sample = [LPS_Net_sample; vertcat(select_sample{non_empty})];
end

% shuffle
rand_idx = randperm(length(LPS_Net_sample));
LPS_Net_sample = LPS_Net_sample(rand_idx, :);
LPS_Net_sample(:,15:16) = []; % BW and SC are fixed so delete

% %%% maybe no need
% % rebuild cdf & 21->16
% num_feat = config.num_feat;
% num_classes = config.num_classes;
% old_num_cdf_sample = config.num_NN_1_cdf_sample;
% new_num_cdf_sample = config.num_NN_2_cdf_sample;
% 
% num_link_level_sample = size(link_level_sample, 1);
% new_link_level_sample = zeros(num_link_level_sample, num_feat+num_classes*new_num_cdf_sample);
% 
% f_old_quatile = [1:50:1000 1000] ./ 1000;
% f_fine = linspace(0, 1000, 1000);
% f_new_quatile = linspace(0, 1, new_num_cdf_sample);
% 
% for sam = 1:num_link_level_sample
%     tmp_sample = link_level_sample(sam, num_feat+1:end); % dim: 1x63
%     tmp_sample = reshape(reshape(tmp_sample', [], 1), old_num_cdf_sample, [])';
%     new_sample = [];
%     for class = 1:size(tmp_sample,1)
%         cdf_interp = interp1(f_old_quatile, tmp_sample(class,:), f_fine, 'pchip');
%         new_sample = [new_sample interp1(f_fine, cdf_interp, f_new_quatile, 'linear')];
%     end
%     new_link_level_sample(sam,:) = [link_level_sample(sam, 1:num_feat) new_sample];
% end

% save train & test sample
idx_arr = 1:config.num_NN_1_cdf_sample;
cdf_direct_name = 'cdf_direct_sigPow_' + string(idx_arr);
cdf_cascade_name = 'cdf_cascade_sigPow_' + string(idx_arr);
cdf_dynamic_name = 'cdf_dynamic_noiPow_' + string(idx_arr);
sample_name = ["ts", "irs_x", "irs_y", "irs_z", "rot_z", "rot_y", "rot_x", ...
               "M_Y", "M_Z", "G_i", "Pa", "Nv_density", "ue_x", "ue_y", ...
               "isBeamform"];
sample_name = [sample_name cdf_direct_name cdf_cascade_name cdf_dynamic_name];

% train&validation : test = 9 : 1
ratio = floor(size(LPS_Net_sample,1) / 10);

train_data_table = array2table(LPS_Net_sample(1:ratio*9,:), 'variableNames', sample_name);
test_data_table = array2table(LPS_Net_sample(ratio*9+1:end,:), 'variableNames', sample_name);

dest_train_path = fullfile(config.vit_save_path, 'train_data.xlsx');
dest_test_path = fullfile(config.vit_save_path, 'test_data.xlsx');

writetable(train_data_table, dest_train_path);
writetable(test_data_table, dest_test_path);

%% NN-2

% same column -> same num of AIRSs
SE_Net_sample = {};
SE_Net_id = 2;
for src_idx = 1:num_src
    UE_cell = load(src_data_path{src_idx}).ue_cell;
    non_empty = ~cellfun(@isempty, UE_cell(:,SE_Net_id));
    SE_Net_sample = [SE_Net_sample; UE_cell(non_empty,SE_Net_id)];
end

% 按照列数分组并拼接
unique_N = unique(cellfun(@(x) size(x, SE_Net_id), SE_Net_sample));
total_train_sample = cell(length(unique_N), 1);
total_test_sample = cell(length(unique_N), 1);

for i = 1:length(unique_N)
    N = unique_N(i);
    matrices_with_N = SE_Net_sample(cellfun(@(x)size(x, SE_Net_id) == N, SE_Net_sample));
    total_data_with_N = vertcat(matrices_with_N{:}); % 将这些矩阵按行拼接
    
    tmp_len = size(total_data_with_N, 1);
    rand_idx = randperm(tmp_len);
    total_data_with_N = total_data_with_N(rand_idx, :);

    ratio = floor(tmp_len/10);
    total_train_sample{i} = total_data_with_N(1:ratio*9, :);
    total_test_sample{i} = total_data_with_N(ratio*9+1:end, :);
end

% %% special process
% 
% num_per_cdf = config.num_NN_1_cdf_sample;
% 
% for i = 1:length(unique_N)
%     tmp_train_sample = total_train_sample{i};
%     valid_id = find(tmp_train_sample(:,42) ~= -600);
%     sel_train_sample = tmp_train_sample(valid_id,:);
%     [num_select, seq_len] = size(sel_train_sample);
% 
%     new_train_sample = zeros(num_select, seq_len-num_per_cdf);
%     for idx = 1:num_select
%         tmp_sam = sel_train_sample(idx, 1:end-1);
%         tmp_sam = reshape(reshape(tmp_sam', [], 1), num_per_cdf, [])';
%         valid_id = find(tmp_sam(:,end) ~= -600);
%         sel_sam = tmp_sam(valid_id,:);
%         sel_sam = reshape(sel_sam', [], 1)';
%         new_train_sample(idx,:) = [sel_sam sel_train_sample(idx, end)];
%     end
% 
%     total_train_sample{i} = new_train_sample;
% end
% 
% for i = 1:length(unique_N)
%     tmp_test_sample = total_test_sample{i};
%     valid_id = find(tmp_test_sample(:,42) ~= -600);
%     sel_test_sample = tmp_test_sample(valid_id,:);
%     [num_select, seq_len] = size(sel_test_sample);
% 
%     new_test_sample = zeros(num_select, seq_len-num_per_cdf);
%     for idx = 1:num_select
%         tmp_sam = sel_test_sample(idx, 1:end-1);
%         tmp_sam = reshape(reshape(tmp_sam', [], 1), num_per_cdf, [])';
%         valid_id = find(tmp_sam(:,end) ~= -600);
%         sel_sam = tmp_sam(valid_id,:);
%         sel_sam = reshape(sel_sam', [], 1)';
%         new_test_sample(idx,:) = [sel_sam sel_test_sample(idx, end)];
%     end
% 
%     total_test_sample{i} = new_test_sample;
% end
% 
% %%-----------------------------------------------%%

%% extra program 21 --> 16

old_num_sample_point = config.num_NN_1_cdf_sample;
new_num_sample_point = config.num_NN_2_cdf_sample;

f_old_quatile = [1:50:1000 1000] ./ 1000;
f_fine = linspace(0, 1, 1000);
f_new_quatile = linspace(0, 1, new_num_sample_point);

old_zero2dBm = config.zero2dBm;
new_zero2dBm = 0;

new_train_result = cell(length(unique_N),1);
for cur = 1:length(total_train_sample)
    tmp_data = total_train_sample{cur};
    tmp_cdf_data = reshape(tmp_data(:,1:end-1)', [], 1);
    num_class = (size(tmp_data, 2) - 1) / old_num_sample_point;
    tmp_cdf_data = reshape(tmp_cdf_data, old_num_sample_point, [])';

    new_cdf_data = zeros(size(tmp_cdf_data,1), new_num_sample_point);
    for cdf = 1:length(tmp_cdf_data)
        old_cdf_data = tmp_cdf_data(cdf,:);
        tmp_cdf_interp = interp1(f_old_quatile, old_cdf_data, f_fine, 'pchip');
        new_cdf_data(cdf,:) = interp1(f_fine, tmp_cdf_interp, f_new_quatile, 'linear'); % interpolation

        % % LOG
        % figure;
        % % plot(cdf_interp, f_fine, 'r-');
        % plot(new_cdf_data(cdf,:), f_new_quatile, 'r-');
        % hold on
        % plot(old_cdf_data, f_old_quatile, 'b-.');
    end

    % calibration
    new_cdf_data(new_cdf_data < old_zero2dBm) = old_zero2dBm;
    new_cdf_data(abs(new_cdf_data - (old_zero2dBm+1)) < 1) = old_zero2dBm;
    new_cdf_data = reshape(reshape(new_cdf_data', [], 1), num_class*new_num_sample_point, [])';

    % find new zero2dBm
    global_cdf_data = sort(reshape(unique(new_cdf_data), [], 1), 'ascend');
    new_zero2dBm = min(new_zero2dBm, global_cdf_data(2));

    new_train_result{cur} = [new_cdf_data tmp_data(:,end)];
end

new_test_result = cell(length(unique_N),1);
for cur =  1:length(total_test_sample)
    tmp_data = total_test_sample{cur};
    tmp_cdf_data = reshape(tmp_data(:,1:end-1)', [], 1);
    num_class = (size(tmp_data, 2) - 1) / old_num_sample_point;
    tmp_cdf_data = reshape(tmp_cdf_data, old_num_sample_point, [])';

    new_cdf_data = zeros(size(tmp_cdf_data,1), new_num_sample_point);
    for cdf = 1:length(tmp_cdf_data)
        old_cdf_data = tmp_cdf_data(cdf,:);
        tmp_cdf_interp = interp1(f_old_quatile, old_cdf_data, f_fine, 'pchip');
        new_cdf_data(cdf,:) = interp1(f_fine, tmp_cdf_interp, f_new_quatile, 'linear'); % interpolation
    end

    % calibration
    new_cdf_data(new_cdf_data < old_zero2dBm) = old_zero2dBm;
    new_cdf_data(abs(new_cdf_data - (old_zero2dBm+1)) < 1) = old_zero2dBm;
    new_cdf_data = reshape(reshape(new_cdf_data', [], 1), num_class*new_num_sample_point, [])';

    % find new zero2dBm
    global_cdf_data = sort(reshape(unique(new_cdf_data), [], 1), 'ascend');
    new_zero2dBm = min(new_zero2dBm, global_cdf_data(2));

    new_test_result{cur} = [new_cdf_data tmp_data(:,end)];
end

% % update zero2dBm
% for cur = 1:length(total_test_result)
% 
%     tmp_result = new_train_result{cur};
%     tmp_result(tmp_result == zero2dBm) = new_zero2dBm - 2;
%     new_train_result{cur} = tmp_result;
% 
%     tmp_result = new_test_result{cur};
%     tmp_result(tmp_result == zero2dBm) = new_zero2dBm - 2;
%     new_test_result{cur} = tmp_result;
% 
% end

% save train & test sample
for idx = 1:size(new_train_result, 1)
    dest_train_path = fullfile(config.cdf_save_path, ['train_SE_data_', num2str(idx), '.mat']);
    dest_test_path = fullfile(config.cdf_save_path, ['test_SE_data_', num2str(idx), '.mat']);

    train_result = new_train_result{idx};
    test_result = new_test_result{idx};

    save(dest_train_path, "train_result");
    save(dest_test_path, "test_result");
end

end
