% created by Bryanttxc, FUNLab

function gen_sys_sample(config, input_file, NN_gen, num_IRS)
% GEN_SYS_SAMPLE cat the sample for NN-1 and NN-2

%% initial sample generation

if config.create_new_input_file == 1
    fprintf("[Info] numIRS=%d: Create input file!\n", num_IRS);
    gen_sys_level_data(config, input_file, num_IRS);
end

if config.is_plot_graph == 1
    plot_sys_graph(input_file, num_IRS, config.root_path);
end

%% NN-1 sample generation

if NN_gen == 1

    % create cal folder
    save_input_path = fullfile(pwd,'net','UESENet','result','data','cal','input');
    if exist(save_input_path, 'dir')
        rmdir(save_input_path, "s");
        mkdir(save_input_path);
    end

    save_output_path = fullfile(pwd, 'net','UESENet','result','data','cal','output');
    if exist(save_output_path, 'dir')
        rmdir(save_output_path, "s");
        mkdir(save_output_path);
    end

    % load data
    SAMPLE = load(input_file);
    General = SAMPLE.General;
    General_sampleMatrix = SAMPLE.General_sampleMatrix;
    new_IRS_sampleMatrix = SAMPLE.new_IRS_sampleMatrix;
    UE_sampleMatrix = SAMPLE.UE_sampleMatrix;
    lab_endIdx = SAMPLE.lab_endIdx;

    num_IRS_feat = config.num_IRS_feat;
    num_IRS_state = config.num_IRS_state; % is_beamform 0/1
    for num_UE_idx = 1:length(UE_sampleMatrix)

        tmp_UE_sampleMatrix = UE_sampleMatrix{num_UE_idx};

        % build NN-1 sample
        for lab_idx = 1:length(lab_endIdx)-1

            tmp_IRS_sampleMatrix = new_IRS_sampleMatrix(1+lab_endIdx(lab_idx):lab_endIdx(lab_idx+1),:);

            num_IRS = General.num_IRS;
            num_UE = length(tmp_UE_sampleMatrix);
            num_IRS_group = size(tmp_IRS_sampleMatrix,1) / num_IRS;

            for sublab_idx = 1:num_IRS_group
    
                % sorted by ascend
                % IRS
                sel_IRS = tmp_IRS_sampleMatrix(1+(sublab_idx-1)*num_IRS : sublab_idx*num_IRS, :);
                IRS_x_id = 1;
                IRS_y_id = 2;
                sel_IRS = sortrows(sel_IRS, [IRS_x_id IRS_y_id], {'ascend', 'ascend'});
                sel_IRS = repmat(reshape(sel_IRS', [], 1), 1, num_UE*num_IRS_state);
                sel_IRS = reshape(reshape(sel_IRS, [], 1), num_IRS_feat, [])';
    
                % UE
                sel_UE = tmp_UE_sampleMatrix;
                sel_UE = reshape(repmat(sel_UE', num_IRS*num_IRS_state, 1), [], 1);
                sel_UE = reshape(sel_UE, 2, [])';
                UE_x_id = 1;
                UE_y_id = 2;
                sel_UE = sortrows(sel_UE, [UE_x_id UE_y_id], {'ascend', 'ascend'});
    
                % State
                sel_state = [0 1];
                sel_state = reshape(repmat(sel_state, num_IRS, 1), [], 1);
                sel_state = repmat(sel_state, num_UE, 1);
    
                % cat-build-save sample
                ts = General_sampleMatrix(3);
                sample_matrix = [ts*ones(size(sel_IRS,1), 1) sel_IRS sel_UE sel_state];
                whole_sample_table = array2table(sample_matrix, 'VariableNames', ...
                                    ["ts", "irs_x", "irs_y", "irs_z", "rot_z", "rot_y", "rot_x", ...
                                    "M_Y", "M_Z", "G_i", "Pa", "Nv_density", "ue_x", "ue_y", ...
                                    "isBeamform"]);
                file_name = ['cal_data_lab_', num2str(lab_idx), '_', num2str(sublab_idx), ...
                              '_numUE_', num2str(length(tmp_UE_sampleMatrix)), '.xlsx'];
                dest_cal_data_path = fullfile(save_input_path, file_name);
                writetable(whole_sample_table, dest_cal_data_path);
    
            end % end sublab_idx
        end % end lab_idx
    end % end num_ue_idx
end % end if

%% NN-2 sample generation

if NN_gen == 2

    vit_load_path = fullfile(pwd, 'net', 'UESENet', 'result', 'data', 'cal', 'input');
    cdf_load_path = fullfile(pwd, 'net', 'UESENet', 'result', 'data', 'cal', 'output');
    cdf_save_path = fullfile(pwd, 'net', 'UESENet', 'result', 'data', 'cal', 'input');

    num_lab = config.num_lab;
    max_num_sublab = config.max_num_sublab;

    SAMPLE = load(input_file);
    General = SAMPLE.General;
    UE_sampleMatrix = SAMPLE.UE_sampleMatrix;

    for num_UE_idx = 1:length(UE_sampleMatrix)

        for lab = 1:num_lab

            for sub_lab = 1:max_num_sublab

                %% load data

                % check if file exists
                file_name = ['cal_data_lab_', num2str(lab), '_', num2str(sub_lab), ...
                                '_numUE_', num2str(length(UE_sampleMatrix{num_UE_idx})), '.xlsx'];
                target_file = fullfile(vit_load_path, file_name);
                if ~exist(target_file, 'file')
                    break;
                end

                % feat_data
                %  1    2     3     4     5     6     7    8   9  10       
                % [ts irs_x irs_y irs_z rot_z rot_y rot_x M_Y M_Z G_i 
                % 11     12      13   14      15
                % Pa Nv_density ue_x ue_y is_beamform];
                feat_path = target_file;
                feat_data = table2array(readtable(feat_path));
                
                file_name = ['cdf_data_lab_', num2str(lab), '_', num2str(sub_lab), ...
                                     '_numUE_', num2str(length(UE_sampleMatrix{num_UE_idx})), '.mat'];
                NN_1_result_path = fullfile(cdf_load_path, file_name);
                cdf_data = load(NN_1_result_path).cdf_data;

                ue_x_id = 13;
                unique_UE = unique(feat_data(:,ue_x_id));
                num_UE = length(unique_UE);
                UE_group = cell(num_UE, 1);
                zero2dBm = config.zero2dBm;
                for ue = 1:num_UE
                    tmp_data = cdf_data(feat_data(:,ue_x_id) == unique_UE(ue), :);
                    tmp_data(tmp_data < zero2dBm) = zero2dBm;
                    tmp_data(abs(tmp_data - (zero2dBm+1)) < 1) = zero2dBm;
                    UE_group{ue} = tmp_data;
                end

                %% build sample
                % sample is in order
                
                num_BS = config.num_BS;
                num_IRS = General.num_IRS;
                num_NN_1_cdf_sample = config.num_NN_1_cdf_sample;
                num_classes = config.num_classes; % dir, cas/sca, dyn
                indices = (1:num_classes) .* num_NN_1_cdf_sample;
    
                % NN_2 sample format:
                % direct link(1) | cascade link(1) | scatter link(num_IRS) | dynamic noise(num_IRS)
                new_NN_2_sample = zeros(num_IRS+num_BS, (2+2*num_IRS)*num_NN_1_cdf_sample);
                UE_cell = cell(num_UE, 1);
                sca_start_pos = 2*num_NN_1_cdf_sample + 1;
    
                for ue = 1:num_UE
    
                    tmp_data = UE_group{ue};
    
                    % direct link
                    dir_link_cdf = mean(tmp_data(:, indices(1)-num_NN_1_cdf_sample+1:indices(1)), 1);
                    % cascade link
                    cas_link_cdf = tmp_data(1:num_IRS, indices(2)-num_NN_1_cdf_sample+1:indices(2));
                    % scatter link
                    sca_link_cdf = tmp_data(num_IRS+1:end, indices(2)-num_NN_1_cdf_sample+1:indices(2));
                    sca_link_cdf = reshape(sca_link_cdf', [], 1)';
                    % dynamic noise
                    dyn_noise_cdf = tmp_data(:, indices(3)-num_NN_1_cdf_sample+1:indices(3));
                    dyn_noise_cdf = reshape(reshape(dyn_noise_cdf', [], 1), num_IRS*num_NN_1_cdf_sample, []);
                    dyn_noise_cdf = mean(dyn_noise_cdf, 2)';
    
                    for irs_id = 1:num_IRS
                        new_NN_2_sample(irs_id,:) = [dir_link_cdf cas_link_cdf(irs_id,:) sca_link_cdf dyn_noise_cdf];
                        new_NN_2_sample(irs_id, sca_start_pos+(irs_id-1)*num_NN_1_cdf_sample : sca_start_pos+irs_id*num_NN_1_cdf_sample-1) = zero2dBm;
                    end
                    new_NN_2_sample(end,:) = [dir_link_cdf zero2dBm*ones(1,num_NN_1_cdf_sample) sca_link_cdf dyn_noise_cdf];
    
                    UE_cell{ue} = new_NN_2_sample;
                end
    
                %% 21 sample points -> 16 sample points
    
                unique_N = unique(cellfun(@(x) size(x,2), UE_cell));
                total_test_result = cell(length(unique_N), 1);
    
                for i = 1:length(unique_N)
                    N = unique_N(i);
                    matrices_with_N = UE_cell(cellfun(@(x) size(x, 2) == N, UE_cell));
                    total_data_with_N = vertcat(matrices_with_N{:}); % 将这些矩阵按行拼接
                    total_test_result{i} = total_data_with_N;
                end
    
                old_num_sample_point = config.num_NN_1_cdf_sample;
                new_num_sample_point = config.num_NN_2_cdf_sample;
                
                f_old_quatile = [1:50:1000 1000] ./ 1000;
                f_fine = linspace(0, 1, 1000);
                f_new_quatile = linspace(0, 1, new_num_sample_point);
    
                new_cal_result = cell(length(unique_N),1);
                new_zero2dBm = 0;
                for cur = 1:length(total_test_result)
                    tmp_data = total_test_result{cur};
                    tmp_cdf_data = reshape(tmp_data', [], 1);
                    num_class = size(tmp_data, 2) / old_num_sample_point;
                    tmp_cdf_data = reshape(tmp_cdf_data, old_num_sample_point, [])';
    
                    new_cdf_data = zeros(size(tmp_cdf_data,1), new_num_sample_point);
                    for cdf = 1:length(tmp_cdf_data)
                        pow_dB = tmp_cdf_data(cdf,:);
                        cdf_interp = interp1(f_old_quatile, pow_dB, f_fine, 'pchip');
                        new_cdf_data(cdf,:) = interp1(f_fine, cdf_interp, f_new_quatile, 'linear'); % interpolation
                        % new_cdf_data(cdf,:) = interp1(f_pow, pow_dB, f_fixed, 'linear', 'extrap'); % interpolation
                    end
    
                    new_cdf_data(new_cdf_data < zero2dBm) = zero2dBm;
                    new_cdf_data = reshape(reshape(new_cdf_data', [], 1), num_class*new_num_sample_point, [])';
                    
                    global_cdf_data = sort(reshape(unique(new_cdf_data), [], 1), 'ascend');
                    new_zero2dBm = min(new_zero2dBm, global_cdf_data(2));

                    new_cal_result{cur} = new_cdf_data;
                end

                % update zero2dBm
                for cur = 1:length(new_cal_result)
                
                    tmp_result = new_cal_result{cur};
                    tmp_result(tmp_result == zero2dBm) = new_zero2dBm - 2;
                    new_cal_result{cur} = tmp_result;
                
                end
    
                %% save as NN-2 sample
    
                for idx = 1:size(new_cal_result, 1)
                    file_name = ['cal_SE_data_', num2str(idx), ...
                                 '_lab_', num2str(lab), '_', num2str(sub_lab), ...
                                 '_numUE_', num2str(length(UE_sampleMatrix{num_UE_idx})), '.mat'];
                    dest_test_path = fullfile(cdf_save_path, file_name);
                    cal_result = new_cal_result{idx};
                    save(dest_test_path, "cal_result");
                end

            end % end sublab
        end % end lab
    end % end num_ue_idx
end % end if
