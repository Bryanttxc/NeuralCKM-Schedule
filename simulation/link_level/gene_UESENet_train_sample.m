% created by Bryanttxc, FUNLab

function gene_UESENet_train_sample(config)
%GENE_UESENET_TRAINSAMPLE generate samples for UESENet's training 
% based on QuaDRiGa platform

% Include paths of other folders
root_path = config.root_path;
addpath(fullfile(root_path, 'util'));
addpath(fullfile(root_path, 'util', 'link_level'));
addpath(fullfile(root_path, 'channel', 'quadriga_src'));
addpath(fullfile(root_path, 'channel', 'qris-tidy', 'class'));
addpath(fullfile(root_path, 'channel', 'qris-tidy', 'func'));

% Turn on parallel pool
turnOnParfor(config);

%% Preparation

% Generate Sample
sample_path = config.load_data_path;
if exist(sample_path, 'file') == 2
    fprintf("file is already existed!\n");
    UESENet_sample = load(sample_path);
    General_sampleMatrix = UESENet_sample.General_sampleMatrix;
    IRS_sampleMatrix = UESENet_sample.IRS_sampleMatrix;
    UE_sampleMatrix = UESENet_sample.UE_sampleMatrix;
    BS = UESENet_sample.BS;
    AIRS_set = UESENet_sample.AIRS_set;
    UE = UESENet_sample.UE;
    Power = UESENet_sample.Power;
    General = UESENet_sample.General;
else
    [General_sampleMatrix, IRS_sampleMatrix, UE_sampleMatrix, ...
     BS, AIRS_set, UE, Power, General] = generator();
    save(sample_path, "General_sampleMatrix", "IRS_sampleMatrix", "UE_sampleMatrix", ...
        "General", "BS", "AIRS_set", "UE", "Power");
end

% config QuaDRiGa environment
smlt_para = qd_simulation_parameters; % simulation parameters
smlt_para.center_frequency = General.freq_center;
smlt_para.sample_density = 4; % 4 samples per half-wavelength, relevant to Nyquist
smlt_para.use_absolute_delays = 1; % Include delay of the LOS path
smlt_para.show_progress_bars = 0; % Disable progress bars
smlt_para.use_3GPP_baseline = 1; % Enable drifting, default = 0

% lab param.
% num of sample
num_IRS_sample = size(IRS_sampleMatrix, 1);
num_UE_sample  = size(UE_sampleMatrix, 1);

num_large_iter = config.num_large_iter; % num of large-scale iteration
num_small_fading = config.num_small_fading; % num of fading realizations

num_AIRS_max = config.num_AIRS_max; % max num of AIRS per UE sample
num_serve_state = config.num_serve_state; % is_beamform = 0/1
num_AIRS_feat = config.num_AIRS_feat;
num_cdf_sample = config.num_NN_1_cdf_sample;
% num_cdf_sample = config.num_NN_2_cdf_sample; % Neural CKM need re-train
num_out_class = config.num_out_class;
num_result = num_cdf_sample * num_out_class;

% other parameters
num_sc_per_RB = General.num_sc_per_RB;
noise_density = Power.noise_density;
trans_power_dBm = Power.trans_power_dBm;

% *final result
% ue_cell{:,1} link_cdf_result
% ue_cell{:,2} sys_SE_result
ue_cell = cell(num_UE_sample, 2);
load(config.save_data_path);

%% Sample Generation

fprintf("[Info] Main Program start...\n");

% % randomly select General samples
% select_gen_idx = randShuffle((1:num_gen_sample), num_gen_max);
% select_gen_sampleMatrix = General_sampleMatrix(select_gen_idx,:);

for ue_sam = 1081:num_UE_sample
    
    % select specific UE
    UE.x = UE_sampleMatrix(ue_sam, 1);
    UE.y = UE_sampleMatrix(ue_sam, 2);
    
    % select specific BW and subcarrier intervals
    bandwidth = 20e6; % select_gen_sampleMatrix(ue_sam, 1);
    sc_interval = 15e3; % select_gen_sampleMatrix(ue_sam, 2);
    num_total_RB = floor(bandwidth ./ (sc_interval .* General.num_sc_per_RB)); % select_gen_sampleMatrix(ue_sam, 3);
    num_sc = num_total_RB * General.num_sc_per_RB; % select_gen_sampleMatrix(ue_sam, 4);
    
    % UE_rho = select_gen_sampleMatrix(ue_sam, 5);
    % num_sc_for_ue = round(UE_rho * num_total_RB) * num_sc_per_RB;
    % UE_bandwidth = num_sc_for_ue * sc_interval;
    
    %%% noise power
    noise_power_per_sc = noise_density * sc_interval;
    
    %%% trans power
    % rng(config.random_seed_ts_pow);
    rand_idx = randi(length(trans_power_dBm), 1, 1);
    select_trans_power_dBm = trans_power_dBm(rand_idx);
    trans_power = 10^(select_trans_power_dBm/10)/1000; % dBm -> W
    trans_power_per_sc = trans_power / num_sc; % per subcarrier
    
    % select random number & indices of AIRSs
    % rng(config.random_seed_num_AIRS);
    num_select_AIRS = randi(num_AIRS_max-1) + 1; % 2-6 AIRSs
    rand_order = randperm(num_IRS_sample);
    select_AIRS_idx = rand_order(1:num_select_AIRS);
    
    select_AIRS_set = cell(num_select_AIRS, 1);
    for idx = 1:num_select_AIRS
        irs_sam = select_AIRS_idx(idx);
        select_AIRS_set{idx} = AIRS_set{irs_sam}; 
    end
    
    fprintf("[Info] Case %d: UE %d, ts_pow: %.2f W, bandwidth: %d MHz, sc_interv: %d KHz, num_RB: %d, num_sc: %d, num_AIRS: %d\n", ...
                ue_sam, ue_sam, trans_power, bandwidth/1e6, sc_interval/1e3, num_total_RB, num_sc, num_select_AIRS);
    
    % ---------------------------- generate QuaDRiGa channel ----------------------------%
    [b, all_Nsub] = gene_quadriga_channel(smlt_para, BS, UE, select_AIRS_set, num_sc);

    tmp_direct_signalPow = zeros(num_small_fading, num_large_iter);
    tmp_cascade_signalPow = zeros(num_select_AIRS, num_small_fading, num_large_iter);
    tmp_scatter_signalPow = zeros(num_select_AIRS, num_small_fading, num_large_iter);
    tmp_dynamic_noisePow = zeros(num_select_AIRS, num_small_fading, num_large_iter);
    ergodic_SE_MC = zeros(num_select_AIRS + 1, num_large_iter); % 1 -> the case that all AIRSs' state = 1(scatter)
    
    % fig1 = figure;
    % fig2 = figure;
    % fig3 = figure;
    % fig4 = figure;
    % for loop = 1:5
    tic
    fprintf("[Info] UE %d: Large-scale condition is running: \n", ue_sam);
    for iter = 1:num_large_iter
        fprintf("[Info] UE %d: Large-scale iter %d\n", ue_sam, iter);
        %---------------------------- Calculate indicators ----------------------------%
        [tmp_direct_signalPow(:,iter), ...
         tmp_cascade_signalPow(:,:,iter), ...
         tmp_scatter_signalPow(:,:,iter), ...
         tmp_dynamic_noisePow(:,:,iter), ...
         ergodic_SE_MC(:,iter)] = func_get_freq_resp(b, num_select_AIRS, select_AIRS_set, 1, 1, ...
                                                   all_Nsub, bandwidth, sc_interval, ...
                                                   num_sc, num_small_fading, ...
                                                   trans_power_per_sc, noise_power_per_sc);
    end
    end_time = toc;
    fprintf("[Info] UE %d: Large-scale condition done! Cost time: %.4f s\n", ue_sam, end_time);

    %---------------------------- generate Emprical CDF ----------------------------%
    select_idx = [1:50:1000 1000];
    
    %%% direct
    dir_sigPow_dB_fixed = gene_ecdf(tmp_direct_signalPow);
    dir_sigPow_dB_select = dir_sigPow_dB_fixed(select_idx);
    
    %%% cascade
    cas_sigPow_dB_select = zeros(num_select_AIRS, num_cdf_sample);
    sca_sigPow_dB_select = zeros(num_select_AIRS, num_cdf_sample);
    dyn_noiPow_dB_select = zeros(num_select_AIRS, num_cdf_sample);
    
    for idx = 1:num_select_AIRS
        
        cas_sigPow_dB_fixed = gene_ecdf(tmp_cascade_signalPow(idx,:,:));
        sca_sigPow_dB_fixed = gene_ecdf(tmp_scatter_signalPow(idx,:,:));
        dyn_noiPow_dB_fixed = gene_ecdf(tmp_dynamic_noisePow(idx,:,:));
        
        cas_sigPow_dB_select(idx,:) = cas_sigPow_dB_fixed(select_idx);
        sca_sigPow_dB_select(idx,:) = sca_sigPow_dB_fixed(select_idx);
        dyn_noiPow_dB_select(idx,:) = dyn_noiPow_dB_fixed(select_idx);
        
    end
    
    %     %------- Emprical CDF LOG (check validation of 150 large + 400 small) -------%
    %     f_fixed = linspace(0, 1, 1000);
    % 
    %     figure(fig1);
    %     hold on
    %     plot(dir_sigPow_dB_select, f_fixed(select_idx), 'r-.'); % direct link
    % 
    %     figure(fig2);
    %     hold on
    %     plot(cas_sigPow_dB_select(1,:), f_fixed(select_idx), 'r-.'); % cascade link
    % 
    %     figure(fig3);
    %     hold on
    %     plot(sca_sigPow_dB_select(1,:), f_fixed(select_idx), 'r-.'); % scatter link
    % 
    %     figure(fig4);
    %     hold on
    %     plot(dyn_noiPow_dB_select(1,:), f_fixed(select_idx), 'r-.'); % dynamic noise
    % end
    
    %---------------------------- assemble result ----------------------------%
    %%% link-level
    tmp_trans_power_dBm = repmat(select_trans_power_dBm, num_serve_state*num_select_AIRS, 1); % is_beamform = 0 or 1
    tmp_irs_sample = reshape( repmat(reshape(IRS_sampleMatrix(select_AIRS_idx,:), 1, []), num_serve_state, 1), [], num_AIRS_feat ); % careful
    tmp_ue_sample = repmat([UE.x, UE.y, bandwidth, sc_interval], num_serve_state*num_select_AIRS, 1);
    tmp_is_beamform = repmat([0;1], num_select_AIRS, 1);
    
    % result
    tmp_dir = repmat(dir_sigPow_dB_select, num_serve_state*num_select_AIRS, 1);
    tmp_cas = reshape(reshape([cas_sigPow_dB_select sca_sigPow_dB_select]', [], 1), num_cdf_sample, [])'; % careful
    tmp_noi = reshape(reshape(repmat(dyn_noiPow_dB_select, 1, num_serve_state)', [], 1), num_cdf_sample, [])'; % careful
    link_result = [tmp_dir tmp_cas tmp_noi];
    
    link_cdf_result = [tmp_trans_power_dBm, tmp_irs_sample, tmp_ue_sample, tmp_is_beamform, link_result];
    
    %%% system-level
    zero2dBm = config.zero2dBm;
    tmp_dir = repmat(dir_sigPow_dB_select, num_select_AIRS+1, 1);
    tmp_noi = repmat(reshape(dyn_noiPow_dB_select', 1, []), num_select_AIRS+1, 1); % careful
    tmp_sca = zeros(num_select_AIRS+1, num_select_AIRS*num_cdf_sample);
    for idx = 1:num_select_AIRS
        pre_sca_idx = 1:idx-1;
        lat_sca_idx = idx+1:num_select_AIRS;
        tmp_sca(idx,:) = [reshape(sca_sigPow_dB_select(pre_sca_idx,:)', 1, []), ...
                          zero2dBm*ones(1, num_cdf_sample), ...
                          reshape(sca_sigPow_dB_select(lat_sca_idx,:)', 1, [])]; % careful
    end
    tmp_sca(end,:) = reshape(sca_sigPow_dB_select', 1, []); % careful

    % result
    rate_result = mean(ergodic_SE_MC, 2); % (num_select_AIRS + 1) x 1
    per_AIRS_serve_result = [tmp_dir(1:end-1,:) cas_sigPow_dB_select tmp_sca(1:end-1,:) tmp_noi(1:end-1,:) rate_result(1:end-1,:)];
    all_AIRS_scatter_result = [tmp_dir(end,:) zero2dBm*ones(1, num_cdf_sample) tmp_sca(end,:) tmp_noi(end,:) rate_result(end,:)];
    system_MC_rate_result = [per_AIRS_serve_result; all_AIRS_scatter_result];

    %%% Final Result
    ue_cell{ue_sam,1} = link_cdf_result;
    ue_cell{ue_sam,2} = system_MC_rate_result;
    
    % save data per 5 sample (prevent accident)
    if mod(ue_sam, 5) == 0
        save(config.save_data_path,'ue_cell');
    end

end % for ue_sam

% Turn off parallel pool if open
if pick_parfor == 1
    delete(gcp('nocreate'));
end

end
