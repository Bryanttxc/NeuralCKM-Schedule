% created by Bryanttxc, FUNLab

function util_tool(src_path_vec)
% UTIL_TOOL build sample for gen_MC_result.m

%% load data

viT_save_path = fullfile('..', '..', 'net', 'UESENet', 'result', 'data');
cdf_save_path = fullfile('..', '..', 'net', 'UESENet', 'result', 'data');

% link-level
link_level_sample = [];
for src_idx = 1:length(src_path_vec)
    ue_cell = load(src_path_vec(src_idx)).ue_cell;

    isEmpty = cellfun(@isempty, ue_cell(:,1)');
    select_ue_cell_row = find(~isEmpty);
    for ue = select_ue_cell_row
        link_level_sample = [link_level_sample; ue_cell{ue,1}];
    end
end
% random mess
rand_order = randperm(length(link_level_sample));
link_level_sample = link_level_sample(rand_order, :);
link_level_sample(:,15:16) = []; % BW and SC are fixed so delete

% system-level
% same column -> same num of AIRSs
system_level = {};
for src_idx = 1:length(src_path_vec)
    ue_cell = load(src_path_vec(src_idx)).ue_cell;

    isEmpty = cellfun(@isempty, ue_cell(:,2)');
    select_ue_cell_row = ~isEmpty;
    system_level = [system_level;ue_cell(select_ue_cell_row,2)];
end

% 按照列数分组并拼接
unique_N = unique(cellfun(@(x) size(x,2), system_level));
total_test_result = cell(length(unique_N),1);

for i = 1:length(unique_N)
    N = unique_N(i);
    matrices_with_N = system_level(cellfun(@(x) size(x, 2) == N, system_level));
    total_data_with_N = vertcat(matrices_with_N{:}); % 将这些矩阵按行拼接
    
    tmp_len = size(total_data_with_N, 1);
    rand_order = randperm(tmp_len);
    total_data_with_N = total_data_with_N(rand_order, :);

    total_test_result{i} = total_data_with_N; % 1
end

%% save
% link level
num_cdf_sample = 21;
number = 1:num_cdf_sample;
cdf_direct_name = 'cdf_direct_sigPow_' + string(number);
cdf_cascade_name = 'cdf_cascade_sigPow_' + string(number);
cdf_dynamic_name = 'cdf_dynamic_noiPow_' + string(number);
sample_name = ["ts", "irs_x", "irs_y", "irs_z", "rot_z", "rot_y", "rot_x", ...
               "M_Y", "M_Z", "G_i", "Pa", "Nv_density", "ue_x", "ue_y", ...
               "isBeamform"];
sample_name = [sample_name cdf_direct_name cdf_cascade_name cdf_dynamic_name];

test_data_table = array2table(link_level_sample, 'variableNames', sample_name);
dest_test_path = fullfile(viT_save_path, 'sys_test_data.xlsx');
writetable(test_data_table, dest_test_path);

% system level
for idx = 1:size(total_test_result, 1)
    dest_test_path = fullfile(cdf_save_path, ['sys_test_data_', num2str(idx), '.mat']);
    test_result = total_test_result{idx};
    save(dest_test_path, "test_result");
end

end
