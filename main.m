% created by Bryanttxc, FUNLab

clc;
clear;
close all;

addpath(fullfile('algorithm', 'phaseOpt'));
addpath(fullfile('algorithm', 'schedule'));
addpath(fullfile('algorithm', 'RCGAlgorithm'));
addpath(fullfile('channel'));
addpath(fullfile('config'));
addpath(fullfile('net'));
addpath(fullfile('result'));
addpath(fullfile('simulation', 'link_level'));
addpath(fullfile('simulation', 'system_level'));
addpath(fullfile('util'));
addpath(fullfile('util', 'plot'));
addpath(fullfile('util', 'link_level'));
addpath(fullfile('util', 'system_level'));

%% Initialization

LoSLoS_scene_start = 0;
LoSNLoS_scene_start = 0;
phase_compare_start = 1;
gen_NN_sample_start = 0;
SE_map_start = 0;

train_UESENet_start = 0;
schedule_start = 0;

%% link-level

% LoSLoS scene
if LoSLoS_scene_start == 1
    [config] = LoSLoS_scene_config();
    LoSLoS_scene(config);
end

% LoSNLoS scene
if LoSNLoS_scene_start == 1
    [config] = LoSNLoS_scene_config();
    LoSNLoS_scene(config);
end

% comparison of phase opt methods
if phase_compare_start == 1
    [config] = phaseOpt_config();
    % tmp_phaseOpt(config);
    % phaseOpt_compare(config);
    plot_link_level_phase(config);
end

% generate Neural CKM sample
if gen_NN_sample_start == 1
    [config] = gen_NN_sample_config();
    % gene_UESENet_train_sample(config);
    stitch_UESENet_sample(config);
end

% generate SE map
if SE_map_start == 1
    [config] = SE_map_config();
    SE_map(config);
end

%% system-level

% train UESENet
net_idx = 1; % 1->NN-1 / 2->NN-2 / 3->MLP / 4->LSTM
if train_UESENet_start == 1
    UESENet_train(net_idx);
end

% schedule multi-AIRS aided multi-user
if schedule_start == 1
    [config] = schedule_config();

    % plot_sys_level_comp(config);

    % angle = 120;
    % for cur_angle = angle
        % config.init_rot_angle = cur_angle;
        schedule(config);
    % end
end
