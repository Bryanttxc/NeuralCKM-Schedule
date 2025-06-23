% created by Bryanttxc, FUNLab

function [config] = gen_link_level_phase_sample(config, NN_gen)
%GEN_LINK_LEVEL_PHASE_SAMPLE generate samples for SE-Net

%% NN-2 sample generation

if NN_gen == 2

    vit_load_path = fullfile(pwd, 'net', 'UESENet', 'result', 'data', 'cal', 'input');
    cdf_load_path = fullfile(pwd, 'net', 'UESENet', 'result', 'data', 'cal', 'output');
    cdf_save_path = fullfile(pwd, 'net', 'UESENet', 'result', 'data', 'cal', 'input');

    %% load data

    lab = 1;
    sub_lab = 1;

    % check if file exists
    file_name = 'cal_data_lab_1_1_numUE_1.xlsx';
    target_file = fullfile(vit_load_path, file_name);
    if ~exist(target_file, 'file')
        return;
    end

    % feat_data
    %  1    2     3     4     5     6     7    8   9  10       
    % [ts irs_x irs_y irs_z rot_z rot_y rot_x M_Y M_Z G_i 
    % 11     12      13   14      15
    % Pa Nv_density ue_x ue_y is_beamform];
    feat_path = target_file;
    feat_data = table2array(readtable(feat_path));
    
    file_name = 'cdf_data_lab_1_1_numUE_1.mat';
    NN_1_result_path = fullfile(cdf_load_path, file_name);
    cdf_data = load(NN_1_result_path).cdf_data;

    ue_x_id = 13;
    ue_y_id = 14;
    unique_UE_x = unique(feat_data(:,ue_x_id));
    unique_UE_y = unique(feat_data(:,ue_y_id));
    num_UE_x = length(unique_UE_x);
    num_UE_y = length(unique_UE_y);
    num_UE = length(unique_UE_x)*length(unique_UE_y);
    UE_group = cell(num_UE, 1);
    zero2dBm = config.zero2dBm;
    for ue_x = 1:num_UE_x
        for ue_y = 1:num_UE_y
            tmp_data = cdf_data(feat_data(:,ue_x_id) == unique_UE_x(ue_x) & feat_data(:,ue_y_id) == unique_UE_y(ue_y), :);
            tmp_data(tmp_data < zero2dBm) = zero2dBm;
            tmp_data(abs(tmp_data + 599.0) < 1) = zero2dBm;
            UE_group{(ue_x-1)*num_UE_y + ue_y} = tmp_data;
        end
    end

    %% build sample
    % sample is in order
    
    num_BS = config.num_BS;
    num_IRS = config.num_IRS;
    num_NN_1_cdf_sample = config.num_NN_1_cdf_sample;
    num_classes = config.num_classes; % dir, cas/sca, dyn
    indices = (1:num_classes) .* num_NN_1_cdf_sample;

    % NN_2 sample format:
    % direct link(1) | cascade link(1) | scatter link(num_IRS-1) | dynamic noise(num_IRS)
    UE_cell = cell(num_UE, 1);

    for ue = 1:num_UE

        tmp_data = UE_group{ue};
        new_NN_2_sample = zeros(5, (2+2*num_IRS)*num_NN_1_cdf_sample); % revised
        for elem_case = 1:5

            dir_link_cdf = mean( tmp_data( 1+(elem_case-1)*num_IRS:2+(elem_case-1)*num_IRS, indices(1)-num_NN_1_cdf_sample+1:indices(1) ));
            cas_link_cdf = tmp_data( 1+(elem_case-1)*num_IRS, indices(2)-num_NN_1_cdf_sample+1:indices(2) );
            
            sca_link_cdf = tmp_data( 2+(elem_case-1)*num_IRS, indices(2)-num_NN_1_cdf_sample+1:indices(2) );
            
            dyn_noise_cdf = tmp_data( 1+(elem_case-1)*num_IRS:2+(elem_case-1)*num_IRS, indices(3)-num_NN_1_cdf_sample+1:indices(3) );
            dyn_noise_cdf = reshape(reshape(dyn_noise_cdf', [], 1), num_IRS*num_NN_1_cdf_sample, []);
            dyn_noise_cdf = dyn_noise_cdf';

            new_NN_2_sample(elem_case,:) = [dir_link_cdf cas_link_cdf zero2dBm*ones(1,num_NN_1_cdf_sample) sca_link_cdf dyn_noise_cdf];
            % new_NN_2_sample(elem_case,:) = [dir_link_cdf cas_link_cdf sca_link_cdf dyn_noise_cdf];
        end

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
    f_mid_quatile = linspace(0, 1, 1000);
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
            cdf_interp = interp1(f_old_quatile, pow_dB, f_mid_quatile, 'pchip');
            new_cdf_data(cdf,:) = interp1(f_mid_quatile, cdf_interp, f_new_quatile, 'linear'); % method 1
            % new_cdf_data(cdf,:) = interp1(f_old_quatile, pow_dB, f_new_quatile, 'linear', 'extrap'); % method 2
        end

        new_cdf_data(new_cdf_data < zero2dBm) = zero2dBm;
        new_cdf_data = reshape(reshape(new_cdf_data', [], 1), num_class*new_num_sample_point, [])';
        
        global_cdf_data = sort(reshape(unique(new_cdf_data), [], 1), 'ascend');
        new_zero2dBm = min(new_zero2dBm, global_cdf_data(2));
        
        new_cal_result{cur} = new_cdf_data;
    end

    % % update zero2dBm
    % for cur = 1:length(new_cal_result)
    %     tmp_result = new_cal_result{cur};
    %     tmp_result(tmp_result == zero2dBm) = new_zero2dBm - 2;
    %     new_cal_result{cur} = tmp_result;
    % end

    %% save as NN-2 sample

    for idx = 1:size(new_cal_result, 1)
        file_name = ['cal_SE_data_', num2str(idx), ...
                     '_lab_', num2str(lab), '_', num2str(sub_lab), ...
                     '_numUE_1.mat'];
        dest_test_path = fullfile(cdf_save_path, file_name);
        cal_result = new_cal_result{idx};
        save(dest_test_path, "cal_result");
    end

end % end if

end
