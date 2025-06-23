% created by Bryanttxc, FUNLab

function [config] = gen_NN_sample_config()
%SYS_CONFIG Neural CKM sample param. config

config.root_path = pwd;
config.link_src_path = fullfile(config.root_path, 'result', 'data', 'link_level', 'net_src_sample');
config.load_data_path = fullfile(config.link_src_path, 'SAMPLE_250117.mat');

config.link_train_path = fullfile(config.root_path, 'result', 'data', 'link_level', 'net_train_sample');
config.save_data_path = fullfile(config.link_train_path, 'withBS_17feat_250117_viT_Sample_link_plus_system_level_1.mat');

config.vit_save_path = fullfile(config.root_path, 'net', 'UESENet', 'result', 'data', 'train', 'vit');
config.cdf_save_path = fullfile(config.root_path, 'net', 'UESENet', 'result', 'data', 'train', 'cdf');

config.num_large_iter = 150; % num of large-scale iteration
config.num_small_fading = 400; % num of small-fading realizations

config.num_AIRS_max = 6; % max num of Active IRSs
config.num_serve_state = 2; % is_beamform = 0/1
config.num_AIRS_feat = 11; % see 输入输出参数表
config.num_feat = 15; % see 输入输出参数表
config.num_NN_1_cdf_sample = 21; % num of cdf sample points of vit
config.num_NN_2_cdf_sample = 16; % num of cdf sample points of cdf
config.num_out_class = 3; % direct_sigPow, cascade/scatter_sigPow, dynamic_noiPow

config.zero2dBm = -600;
config.random_seed_ts_pow = 2; % fix transmit power
config.random_seed_num_AIRS = 1; % fix num of AIRS in environment

config.pick_parfor = 0; % choose parfor
config.num_workers = 10; % num of parfor workers

end
