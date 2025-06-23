% created by Bryanttxc, FUNLab

function [config] = gen_MC_result_config()
%SYS_CONFIG system-level param. config

config.root_path = pwd;
config.sys_data_path = fullfile(config.root_path, 'result', 'data', 'system_level');
config.schedule_src_path = fullfile(config.sys_data_path, 'schedule_src_sample');
config.load_data_path = fullfile(config.schedule_src_path, 'sys_level_data.mat');
config.save_data_path = fullfile(config.schedule_src_path, 'sys_level_cal.mat');

config.num_UE = 24;
config.num_state = 2; % is_beamform = 0 or 1
config.num_large_iter = 150; % num of large-scale iteration
config.num_small_fading = 400; % num of fading realizations
config.num_IRS_feat = 11; % see 输入输出参数表
config.num_cdf_sample = 21;

config.num_IRS = 6; % number of Active IRSs
config.zero2dBm = -600;
config.num_cdf_sample = 21; % number of cdf sample points

config.pick_parfor = 0; % choose parfor
config.num_workers = 10; % number of parfor workers

end
