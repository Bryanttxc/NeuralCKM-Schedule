% created by Bryanttxc, FUNLab

function [gen_SE_time] = gen_sys_SE_matrix(num_UE, num_served, config)
%GEN_SE_MATRIX generate SE_matrix

num_cdf_sample = config.num_NN_2_cdf_sample;
num_lab = config.num_lab;
max_num_sublab = config.max_num_sublab;
save_data_path = config.save_data_path;
src_SE_path = config.src_SE_path;
zero2dBm = config.zero2dBm;

tic
for lab = 1:num_lab
    for sub_lab = 1:max_num_sublab

        % check if exist SE_matrix
        full_SE_matrix_path = fullfile(save_data_path, ['SE_matrix_lab_', num2str(lab), '_', num2str(sub_lab), ...
                                '_numUE_', num2str(num_UE), '.mat']);
        if exist(full_SE_matrix_path, 'file')
            continue;
        end

        % load data
        full_SE_path = fullfile(src_SE_path, ['SE_data_lab_', num2str(lab), '_', num2str(sub_lab), ...
                        '_numUE_', num2str(num_UE), '.mat']);
        if ~exist(full_SE_path, 'file')
            break;
        end

        % Segment data for each UE
        SE_data = load(full_SE_path).SE_data;
        ue_cell = cell(num_UE, 1);
        for idx = 1:num_UE
            ue_cell{idx,1} = SE_data(1+(idx-1)*num_served:idx*num_served, :);
        end
        
        % build SE_matrix
        SE_matrix = zeros(num_UE, num_served);
        for ue_id = 1:num_UE
            tmp_ue = ue_cell{ue_id, 1};
            [~, sample_dim] = size(tmp_ue);

            goal_dim = 1; % last one is SE
            zero2dBm = min(tmp_ue(:));
            each_cdf_endIdx = num_cdf_sample:num_cdf_sample:sample_dim-goal_dim;
            [served_IRS_pos_x, served_IRS_pos_y] = find(tmp_ue(:,each_cdf_endIdx) == zero2dBm); % served label

            for cur = 1:num_served
                IRS_id = served_IRS_pos_y(cur) - 2; % 2->delete direct link & cascaded link
                if IRS_id  % IRS-served
                    SE_matrix(ue_id, IRS_id) = tmp_ue(served_IRS_pos_x(cur),end);
                else % BS-served
                    SE_matrix(ue_id, end) = tmp_ue(served_IRS_pos_x(cur),end);
                end
            end
        end

        % save data
        save(full_SE_matrix_path, "SE_matrix")
    end
end
gen_SE_time = toc;

end
