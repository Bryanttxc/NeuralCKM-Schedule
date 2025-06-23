% created by Bryanttxc, FUNLab

function phaseOpt_compare(config)
%PHASEOPT_COMPARE compare different phase optimiztion schemes
% at link-level based on QuaDRiGa platform

% include paths of other folders
root_path = config.root_path;
addpath(fullfile(root_path, 'algorithm'));
addpath(fullfile(root_path, 'util'));
addpath(fullfile(root_path, 'util', 'link_level'));
addpath(fullfile(root_path, 'channel', 'quadriga_src'));
addpath(fullfile(root_path, 'channel', 'qris-tidy', 'func'));

% turn on parallel pool
turnOnParfor(config);

%% Preparation

% generate sample
sample_path = config.load_data_path;
[BS, AIRS, UE, Power, General] = gen_phaseOpt_sample(config);
save(sample_path, "BS", "AIRS", "UE", "Power", "General");

% add 2-D position of BS & UE
BS.x = config.BS_x;
BS.y = config.BS_y;

UE.x = config.UE_x;
UE.y = config.UE_y;

% create AIRS_set struct
num_AIRS = config.AIRS_num;
AIRS_set = cell(num_AIRS, 1); % equal 2
for cnt = 1:num_AIRS
    AIRS.x = config.AIRS_x_set(cnt);
    AIRS.y = config.AIRS_y_set(cnt);
    AIRS_set{cnt} = AIRS;
end

% config QuaDRiGa environment
smlt_para = qd_simulation_parameters; % simulated-param.
smlt_para.center_frequency = General.freq_center;
smlt_para.sample_density = 4; % 4 samples per half-wavelength, relevant to Nyquist
smlt_para.use_absolute_delays = 1; % Include delay of the LOS path
smlt_para.show_progress_bars = 0; % Disable progress bars
smlt_para.use_3GPP_baseline = 1; % Enable drifting, default = 0

% lab parameters
num_method = config.num_method;
num_large_iter = config.num_large_iter;
num_small_fading = config.num_small_fading;

trans_power_per_sc = Power.trans_power_per_sc;
noise_power = Power.noise_power;
dynamicNoise_power = Power.dynamicNoise_power;

light_speed = General.light_speed;
bandwidth = General.bandwidth;
sc_interval = General.sc_interval;
num_sc = General.num_sc;
freq_set = General.freq_set;
freq_center = General.freq_center;

M_Y_sets = config.M_Y_sets;
M_Z_sets = config.M_Z_sets;

%% Phase Optimization

for elem_idx = 1:length(M_Y_sets)-1

    tmp_M_Y = M_Y_sets(elem_idx);
    tmp_M_Z = M_Z_sets(elem_idx);

    % override arg.
    for cnt = 1:num_AIRS
        tmp_AIRS = AIRS_set{cnt};
        tmp_AIRS.M_Y = tmp_M_Y;
        tmp_AIRS.M_Z = tmp_M_Z;
        tmp_AIRS.M = tmp_AIRS.M_Y * tmp_AIRS.M_Z;
        tmp_AIRS.element_y_indices = 1-(tmp_AIRS.M_Y+1)/2:(tmp_AIRS.M_Y-1)/2;
        tmp_AIRS.element_z_indices = 1-(tmp_AIRS.M_Z+1)/2:(tmp_AIRS.M_Z-1)/2;
        AIRS_set{cnt} = tmp_AIRS;
    end

    % as an example
    AIRS = AIRS_set{1};

    % layout generate
    RotateMatrix = calRotateMatrix(1, AIRS.zrot, AIRS.yrot, AIRS.xrot);
    [BS, AIRS, UE, ~, Direction] = gen_layout(BS, AIRS, UE, RotateMatrix);

    %% LoS scheme

    % ARV
    [ARV_inc, ARV_ref, ~, ~] = gen_ARV(AIRS, Direction, light_speed, num_sc, freq_set, RotateMatrix);

    % AIRS phase compensate phase shifts of center frequency
    fc_index = round(num_sc/2);
    phi_LoS = exp(-1j*(angle(ARV_inc(:,fc_index)) + angle(ARV_ref(:,fc_index)) ));

    %% MC simulation by instantenous rate

    ergodic_SE_MC = zeros(num_large_iter, num_method);
    ergodic_time = zeros(num_large_iter, 2);

    AIRS_M = AIRS.M;
    AIRS_amplify_power = AIRS.amplify_power;

    for iter = 1:num_large_iter

        fprintf("[Info] Case %d: AIRS Element %d x %d, No.%d channel realization\n", ...
                        elem_idx, tmp_M_Y, tmp_M_Z, iter);

        SE_Theory_MCCM_SVD = zeros(num_small_fading, 1);
        SE_MC_zsw = zeros(num_small_fading, 1);
        SE_MC_MCCM_SVD = zeros(num_small_fading, 1);
        SE_MC_LoS = zeros(num_small_fading, 1);
        SE_MC_random = zeros(num_small_fading, 1);
        time_arr = zeros(num_small_fading, 2);

        % instantenous CSI
        [b, NSubcar] = gene_quadriga_channel(smlt_para, BS, UE, AIRS_set, num_sc);

        % large-scale param.
        gen_parameters(b,4); % usage == 4 (default)

        % small-scale param.
        for numf = 1:num_small_fading

            gen_parameters(b,2); % usage == 2
            c = b.get_channels;
            c_TxRx = c(1);
            c_TxRIS = c(2:1+num_AIRS);
            c_RISRx = c(1+num_AIRS+1:end);

            % dimension: RxNAnt x TxNAnt x NSubcar x num_fading

            % direct link
            H_TxRx_f = c_TxRx(1,1).fr(bandwidth, NSubcar); % 1 x 1 x num_sc
    
            % cascaded link (serving IRS)
            H_TxRIS1_f = c_TxRIS(1,1).fr(bandwidth, NSubcar); % M x 1 x num_sc
            H_RIS1Rx_f = c_RISRx(1,1).fr(bandwidth, NSubcar); % 1 x M x num_sc
            sum_norm_T = trans_power_per_sc .* sum(abs(squeeze(H_TxRIS1_f)).^2, 1); % num_sc x 1
            amplify_factor_inst_1 = sqrt( AIRS_amplify_power ./ ...
                                         (sum(sum_norm_T) + AIRS_M *dynamicNoise_power) );
            % amplify_factor_inst_1 = sqrt( AIRS_amplify_power ./ ...
            %                             (sum(sum_norm_T) + AIRS_M*num_sc*dynamicNoise_power) ); % Neural CKM needs re-train
            
            %%% Adopted: MCCM-based & instant CSI (JiangTao WCNC) %%%
            tic
            [phi_MCCM_SVD, instant_MCCM_1] = mccm_svd_algorithm(H_TxRIS1_f, H_RIS1Rx_f);
            time_mccm = toc;

            % scattered link (non-serving IRS)
            H_TxRIS2_f = c_TxRIS(1,2).fr(bandwidth, NSubcar); % M x 1 x num_sc
            H_RIS2Rx_f = c_RISRx(1,2).fr(bandwidth, NSubcar); % 1 x M x num_sc
            sum_norm_T = trans_power_per_sc .* sum(abs(squeeze(H_TxRIS2_f)).^2, 1); % num_sc x 1
            amplify_factor_inst_2 = sqrt( AIRS_amplify_power ./ ...
                                         (sum(sum_norm_T) + AIRS_M * dynamicNoise_power) );
            % amplify_factor_inst_2 = sqrt( AIRS_amplify_power ./ ...
            %                              (sum(sum_norm_T) + AIRS_M*num_sc*dynamicNoise_power) ); % Neural CKM needs re-train
            phi_scatter = exp(-1j*(2*pi*rand(AIRS_M,1)));

            H_TxRIS2_f = gpuArray(H_TxRIS2_f);
            H_RIS2Rx_f = gpuArray(H_RIS2Rx_f);
            instant_CCM_2 = permute(H_RIS2Rx_f, [2,1,3]) .* (H_TxRIS2_f .* permute(conj(H_TxRIS2_f), [2,1,3])) .* conj(H_RIS2Rx_f);
            instant_MCCM_2 = mean(instant_CCM_2, 3);
            instant_MCCM_2 = double(gather(instant_MCCM_2));

            %%% Benchmark 1: LoS beamforming %%%
            phi_LoS;

            %%% Benchmark 2: Random phase %%%
            phi_random = exp(-1j*(2*pi*rand(AIRS_M,1)));

            % MC simulation
            tmp_d = squeeze(H_TxRx_f); % num_sc x 1

            tmp_i1 = squeeze(H_TxRIS1_f); % M x num_sc
            tmp_r1 = squeeze(H_RIS1Rx_f); % M x num_sc
            
            tmp_i2 = squeeze(H_TxRIS2_f); % M x num_sc
            tmp_r2 = squeeze(H_RIS2Rx_f); % M x num_sc

            % common variable
            func_signalPow_mat = @(phi_beam, phi_scat)trans_power_per_sc .* abs(tmp_d ...
                                                   + amplify_factor_inst_1 .* diag((tmp_r1 .* phi_beam).' * tmp_i1)...
                                                   + amplify_factor_inst_2 .* diag((tmp_r2 .* phi_scat).' * tmp_i2)).^2; % num_sc x 1
            noisePart_mat = sum(abs(tmp_r1).^2, 1)' .* amplify_factor_inst_1^2 .* dynamicNoise_power + ...
                            sum(abs(tmp_r2).^2, 1)' .* amplify_factor_inst_2^2 .* dynamicNoise_power + ...
                            noise_power; % num_sc x 1
            func_SE_MC = @(signalPart_mat)mean( log2(1 + signalPart_mat ./ noisePart_mat), 1);

            %------ Benchmark 3: MCCM-SVD with approximation formula ------%
            mean_direct_power = mean(abs(tmp_d).^2);
            mean_cascade_power = abs(amplify_factor_inst_1^2 * phi_MCCM_SVD.' * instant_MCCM_1 * conj(phi_MCCM_SVD));
            mean_scatter_power = abs(amplify_factor_inst_2^2 * phi_scatter.' * instant_MCCM_2 * conj(phi_scatter));
            
            mean_signal_power = trans_power_per_sc .* (mean_direct_power + mean_cascade_power + mean_scatter_power);
            mean_noise_power = mean(sum(abs(tmp_r1).^2, 1)) .* amplify_factor_inst_1^2 .* dynamicNoise_power + ...
                               mean(sum(abs(tmp_r2).^2, 1)) .* amplify_factor_inst_2^2 .* dynamicNoise_power + ...
                               noise_power;
            SE_Theory_MCCM_SVD(numf,1) = log2(1 + mean_signal_power / mean_noise_power);

            %------ method: MCCM-based ------%
            signalPart_mat = func_signalPow_mat(phi_MCCM_SVD, phi_scatter);
            SE_MC_MCCM_SVD(numf,1) = func_SE_MC(signalPart_mat);

            %------ method: LoS beamforing ------%
            signalPart_mat = func_signalPow_mat(phi_LoS, phi_scatter);
            SE_MC_LoS(numf,1) = func_SE_MC(signalPart_mat);

            %------ method: Random phase------%
            signalPart_mat = func_signalPow_mat(phi_random, phi_scatter);
            SE_MC_random(numf,1) = func_SE_MC(signalPart_mat);

            %------ method: zhangshuowen ------%
            % [SE_MC_zsw(numf,1), time_zsw] = inst_phase_opt_algorithm(Power, H_TxRx_f, H_TxRIS_f, H_RISRx_f, ...
            %                                 amplify_factor_inst, phi_MCCM_SVD, iter, numf);
            % time_arr(numf,:) = [time_zsw, time_mccm];

        end % end numf

        %%% debug
        % fprintf("[Info] SE of Theory scheme: %.4f\n", mean(SE_Theory_MCCM_SVD));
        % fprintf("[Info] SE of MCCM-based scheme: %.4f\n", mean(SE_MC_MCCM_SVD));
        
        % table compare
        % SE_MC = [SE_MC_zsw, SE_MC_MCCM_SVD];
        % ergodic_time(iter,:) = mean(time_arr, 1);

        % SE compare with element increasing
        SE_MC = [SE_Theory_MCCM_SVD, SE_MC_MCCM_SVD, SE_MC_LoS, SE_MC_random];
        % SE_MC = [SE_MC_zsw, SE_Theory_MCCM_SVD, SE_MC_MCCM_SVD, SE_MC_LoS, SE_MC_random]; % num_method = 5
        ergodic_SE_MC(iter,:) = mean(SE_MC, 1);

    end % end for iter

    %% save data

    avg_ergodic_SE = mean(ergodic_SE_MC);
    fprintf("SE: %f, %f, %f\n", avg_ergodic_SE(1), avg_ergodic_SE(2), avg_ergodic_SE(3));

    if freq_center/1e9 ~= floor(freq_center/1e9)
        diff = freq_center/1e9 - floor(freq_center/1e9);
        fc_str = [num2str(floor(freq_center/1e9)), 'p', strrep(num2str(diff), '0.', '')];
    else
        fc_str = General.freq_center/1e9;
    end
    
    % ergodic_SE_MC_Table = array2table(ergodic_SE_MC, 'variableNames', ...
            % {'Zhangshuowen', 'SE_Theory_MCCM_SVD', 'MCCM-SVD', 'LoS', 'random'});
    ergodic_SE_MC_Table = array2table(ergodic_SE_MC, 'variableNames', ...
            {'SE_Theory_MCCM_SVD', 'MCCM-SVD', 'LoS', 'random'});
    path = ['link_level_SE_sc_', num2str(num_sc), ...
            '_from_AIRS_', num2str(M_Y_sets(elem_idx)), 'x', num2str(M_Z_sets(elem_idx)), ...
            '_BW_', num2str(bandwidth/1e6), 'MHz_SC_', num2str(sc_interval/1e3), 'KHz_fc_', fc_str, 'GHz_approx', '.mat'];
    save(fullfile(config.save_data_path, path), 'ergodic_SE_MC_Table', 'ergodic_time');

end

end
