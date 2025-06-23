% created by Bryanttxc, FUNLab

function [config] = SE_map_config()
%SE_MAP_CONFIG link-level param. config

%% gen_link_level_data.m

% sample
config.freq_center = 3.5e9; % GHz
config.bandwidth = 20e6; % MHz
config.subcarrier_space = 15e3; % KHz
config.trans_power = 30; % dBm

% IRS
config.num_IRS_vec = 6; % number of Active IRSs

config.init_radius = 190;
config.init_IRS_z = 10;
config.init_rot_angle = 180;
config.init_M_Y = 12;
config.init_M_Z = config.init_M_Y;
config.init_G_i = 4;
config.init_Pa = 10; % dBm
config.init_Nv_density = -160; % dBm/Hz

% UE
config.init_UE_radius = 248; % m
config.interval = 5;

%% gen_sample.m

% create input_file
config.is_plot_graph = 1;
config.create_new_input_file = 1;

% NN-1 sample generation
config.num_IRS_feat = 11;
config.num_IRS_state = 2; % is_beamform 0/1

% NN-2 sample generation
config.zero2dBm = -600;

% build sample
config.num_BS = 1; % number of BS
config.num_NN_1_cdf_sample = 21; % number of cdf sample points
config.num_NN_2_cdf_sample = 16;
config.num_classes = 3;

%% SE_map.m

config.radius = linspace(25,250,6); % 6 cases

config.num_lab = 1;
config.max_num_sublab = 1;

config.pick_parfor = 0; % choose parfor
config.num_workers = 2; % number of parfor workers

config.root_path = pwd;
config.link_data_path = fullfile(config.root_path, 'result', 'data', 'link_level', 'SEMap');
config.src_SE_path = fullfile(config.root_path, 'net', 'UESENet', 'result', 'data', 'cal', 'output');
config.SE_map_src_path = fullfile(config.link_data_path, 'SE_src_sample');
config.save_data_path = fullfile(config.link_data_path, 'SE_src_sample');

config.py_interpreter = 'C:\Users\Sherlock\.conda\envs\UESENet\python.exe';
config.py_uesenet_main_path = fullfile(config.root_path, 'net', 'UESENet', 'main.py');

%% plot_SE_map.m

config.fontsize = 30;
config.fontname = 'Times New Roman';

config.save_fig_path = fullfile(config.root_path, 'result', 'figure', 'link_level', 'SEMap');

end
