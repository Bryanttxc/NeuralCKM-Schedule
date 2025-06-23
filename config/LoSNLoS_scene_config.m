% created by Bryanttxc, FUNLab

function [config] = LoSNLoS_scene_config()
%SYS_CONFIG link-level param. config

config.root_path = pwd;
config.load_data_path = fullfile(config.root_path, 'result', 'data', 'link_level');
config.save_data_path = fullfile(config.load_data_path, 'LoSNLoS');

config.num_method = 3;
config.num_angleTrial = 30;
config.num_small_fading = 500;
config.rand_seed = 5;

config.pick_parfor = 1; % choose parfor
config.num_workers = 2; % number of parfor workers

config.debug = 0; % turn off debug
config.scene = 2; % pick scene 2
config.save_rcg_data = 1;

config.plot_powGain = 0; % no plot powGain

config.linewidth = 2;
config.linestyle = '-';
config.fontsize = 28;
config.fontname = 'Times New Roman';
config.color = ["r", "g", "k", "b"];

%% oneBS_oneIRS_oneUE_scene.m

% band
% 4G standard: freq_center->2GHz, BW->20MHz
% 5G standard: freq_center->3.5GHz, BW->100MHz
config.light_speed = 3e8; % m/s
config.freq_center = 2e9; % GHz
config.bandwidth = 100e6; % MHz
config.num_sc = 129; % number of subcarriers

% multipath
config.num_path_BStoUE = 15;
config.num_path_AIRStoUE = 15;
config.max_delay = 20e-9; % ns

% AIRS element & subcarrier
config.M_Y_sets = 8:2:16;
config.M_Z_sets = 8:2:16;

index = 2:8;
config.num_sc_sets = 2.^index + 1; % subcarriers

% power
config.noise_density_dBmPerHz = -174; % dBm/Hz
config.noise_figure = 6; % dB
config.trans_power_dBm = 10; % dBm

% position
config.BS_z = 25; % m
config.UE_z = 1.5;

% pathloss
config.const_PL_BStoUE = 3;
config.const_PL_BStoAIRS = 2;
config.const_PL_AIRStoUE = 2.5;

% AIRS
config.num_AIRS = 1;
config.AIRS_antenna_gain = 4;
config.AIRS_dynamicNoise_density = -160; % dBm/Hz
config.AIRS_M_Y = 16;
config.AIRS_z = 10;
config.AIRS_horizon_rot_angle = 120;

end
