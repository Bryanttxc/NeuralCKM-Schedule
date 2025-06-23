% created by Bryanttxc, FUNLab

function [config] = schedule_config()
%SCHEDULE_CONFIG system-level param. config

%% schedule.m

% path
config.root_path = pwd;
config.sys_data_path = fullfile(config.root_path, 'result', 'data', 'system_level');
config.save_data_path = fullfile(config.sys_data_path, 'tmp');
config.schedule_src_path = fullfile(config.sys_data_path, 'schedule_src_sample');

config.py_interpreter = 'C:\Users\Sherlock\.conda\envs\UESENet\python.exe'; % depend on your device
config.py_UESENet_main_path = fullfile(config.root_path, 'net', 'UESENet', 'main.py');
config.py_Kmeans_path = fullfile(config.root_path, 'util', 'system_level', 'KmeansConstr.py');

config.src_SE_path = fullfile(config.root_path, 'net', 'UESENet', 'result', 'data', 'cal', 'output');

% switch
config.pick_parfor = 0; % choose parfor
config.print_opt_result = 0; % print optimized result

config.use_bench_4_1_random = 0;
config.use_bench_4_1_sort = 0;

config.use_bench_4_2_enum = 0;
config.use_bench_4_2_greedy = 0;
config.use_bench_4_2_Hungarian = 0;

config.use_bench_4_3_greedy = 0;
config.use_bench_4_3_enhanced = 0;
config.use_4_3_enhanced = 1;

% default param.
config.num_workers = 2; % number of parfor workers
config.seed = 4; % random seed for UE positions
config.converge_gap = 1e-4;
config.phase1_tolerate = 1;
config.phase2_tolerate = 3;
config.N_max = 500;

%% gen_sys_level_data.m

% sample
config.freq_center = 3.5e9; % GHz

config.num_IRS_vec = 6; % number of Active IRSs
config.num_slot = 5; % number of time slot
% config.num_UE_vec = (30:config.num_slot:35)'; % number of UE
% config.num_UE_vec = 36; % number of UE
config.num_UE_vec = linspace(30, 210, 10);

config.num_lab = 1; % number of labs
config.max_num_sublab = 30; % upper bound of number may not reach

config.bandwidth = 20e6; % MHz
config.subcarrier_space = 15e3; % KHz
config.trans_power = 30; % dBm

% lab
config.init_radius = 190;
config.init_IRS_z = 10;
config.init_rot_angle = 120;
config.init_M_Y = 12;
config.init_M_Z = config.init_M_Y;
config.init_G_i = 4;
config.init_Pa = 10;
config.init_Nv_density = -160;
config.init_UE_radius = 248;

% lab 1
config.element_vec = 8:2:16;
% lab 2
config.horizon_dist_vec = 10:60:250;
% config.horizon_dist_vec = linspace(25,250,6);
% lab 3
config.height_vec = 5:10:35;
% lab 4
config.rotate_vec = 90:30:240;
% lab 5
q_vec = 1:1:5;
Gi_vec = 2*(q_vec+1);
config.Gi_vec = round(10*log10(Gi_vec));

%% gen_sample.m

% create input_file
config.is_plot_graph = 0;
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

%% plot_result.m

config.linewidth = 5;
config.linestyle = '-';
config.markerSize = 15;
config.fontsize = 20;
config.fontname = 'Times New Roman';
config.three_color = [252,141,89; 255,255,191; 145,207,96] ./ 255;

config.four_color_choice1 = [202,0,32; 244,165,130; 186,186,186; 64,64,64] ./ 255;
config.four_color_choice2 = [215,25,28; 253,174,97; 171,217,233; 44,123,182] ./ 255;
config.four_color_choice3 = [215,25,28; 253,174,97; 171,221,164; 43,131,186] ./ 255;
config.four_color_choice4 = [230,97,1; 253,184,99; 178,171,210; 94,60,153] ./ 255;

config.five_color = [215,25,28; 253,174,97; 255,255,191; 171,221,164; 43,131,186] ./ 255; 

%% plot_sys_level_comp

config.three_color_comp = [228,26,28; 55,126,184; 77,175,74] ./ 255;

end
