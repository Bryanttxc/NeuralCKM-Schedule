% created by Bryanttxc, FUNLab

function [config] = phaseOpt_config()
%SYS_CONFIG link-level parameters config

config.freq_center = 3.5e9; % GHz
config.bandwidth = 20e6; % MHz
config.sc_interval = 15e3; % subcarrier space, numerology 1->15KHz

config.BS_x = 0;
config.BS_y = 0;

config.trans_power_dBm = 10; % dBm
config.amplify_power_dBm = 20; % dBm
% config.AIRS_x_set = [80 80];
% config.AIRS_y_set = [0 -100];
config.AIRS_x_set = [132.320227134890 -100.934326905497];
config.AIRS_y_set = [-243.864654701392 -179.420368676335];
config.AIRS_height = [3.95380611903956 12.8508469603278]; % m
config.AIRS_max_ERP = [10 6]; % 6 dBi
config.AIRS_Nv_density = [-152.969692337842 -150.069926480617]; % dynamic noise power, dBm/Hz
config.AIRS_num = length(config.AIRS_x_set);

config.AIRS_zrot = [140.557736608012 38.6305673971921];
config.AIRS_yrot = [-82.1284110257923 -163.644816652621];
config.AIRS_xrot = [66.7905320884512 160.642797084774];

config.AIRS_M_Y = 8;
config.AIRS_M_Z = config.AIRS_M_Y; % square

config.UE_x = 226.796643346481; % m
config.UE_y = 87.1257933792919;

config.M_Y_sets = 8:2:16;
config.M_Z_sets = config.M_Y_sets; % square

config.root_path = pwd;
config.link_data_path = fullfile(config.root_path, 'result', 'data', 'link_level');
config.save_data_path = fullfile(config.link_data_path, 'phaseComp');
config.load_data_path = fullfile(config.save_data_path, 'LoSNLoS_QuaDRiGa.mat');

config.py_interpreter = 'C:\Users\Sherlock\.conda\envs\UESENet\python.exe';
config.py_uesenet_main_path = fullfile(config.root_path, 'net', 'UESENet', 'main.py');

config.num_method = 4;
config.num_large_iter = 150;
config.num_small_fading = 400;

config.pick_parfor = 1; % choose parfor
config.num_workers = 10; % number of parfor workers

config.num_BS = 1;
config.num_IRS = 2;
config.num_NN_1_cdf_sample = 21;
config.num_NN_2_cdf_sample = 16;
config.num_classes = 3;
config.zero2dBm = -600;

config.markersize = 20;
config.linewidth = 3;
config.fontsize = 30;
config.fivecolor = [202,0,32; 
                    244,165,130; 
                    146,197,222; 
                    5,113,176; 
                    64,64,64] ./ 255;

end
