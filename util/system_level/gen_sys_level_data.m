% created by Bryanttxc, FUNLab

function gen_sys_level_data(config, input_file, num_IRS)
% GEN_SYS_LEVEL_DATA generate the basics of samples and save in save_data_path
%
% %%% input %%%
% save_data_path: save path
% seed: random seed

addpath(fullfile(config.root_path, 'util', 'link_level'));

General.light_speed = 299792458; % exactly light speed, same as quadriga
General.freq_center = config.freq_center; % central frequency
General.wavelength_center = General.light_speed / General.freq_center; % m
General.num_sc_per_RB = 12; % num of subcarriers per RB

General.num_IRS = num_IRS;
General.num_lab = config.num_lab;
General.num_slot = config.num_slot;

noise_density_dBmPerHz = -174; % dBm/Hz
noise_figure = 6; % dB
General.noise_density = 10^((noise_density_dBmPerHz + noise_figure)/10)/1000; % dBm/Hz->W/Hz

BW = config.bandwidth; % fixed
SC = config.subcarrier_space;
ts = config.trans_power;
General_sampleMatrix = [BW; SC; ts];

%% BS

BS.x = 0;
BS.y = 0;
BS.z = 25; % m

BS.Antennas = 1;

%% IRS

% -- 初始状态
num_irs = General.num_IRS;
radius = config.init_radius;
angle = (0: 360/num_irs: 359)';
irs_x = radius .* cosd(angle);
irs_y = radius .* sind(angle);
irs_z = repmat(config.init_IRS_z, num_irs, 1);

% rotation
rot_angle = config.init_rot_angle;
rot_z = mod((0:num_irs-1)' * (360/num_irs) + rot_angle, 360);
rot_z(rot_z > 180) = rot_z(rot_z > 180) - 360; % range: -180~180
rot_y = zeros(num_irs, 1);
rot_x = zeros(num_irs, 1);

M_Y = config.init_M_Y * ones(num_irs, 1);
M_Z = config.init_M_Z * ones(num_irs, 1);
G_i = config.init_G_i * ones(num_irs, 1); % 10log10(4) = 6 dBi
Pa = config.init_Pa * ones(num_irs, 1);
Nv_density = config.init_Nv_density * ones(num_irs, 1);

%                          1     2     3     4     5     6    7   8   9  10     11
base_IRS_sampleMatrix = [irs_x irs_y irs_z rot_z rot_y rot_x M_Y M_Z G_i Pa Nv_density];

IRS_x_id = 1;
IRS_y_id = 2;

%% Labs setup for IRS

new_IRS_sampleMatrix = [];
lab_endIdx = [0];

% % lab1 -- 单元数
% elem_cand = config.element_vec;
% tmp_matrix = [];
% for m = 1:length(elem_cand)
%     tmp = base_IRS_sampleMatrix;
%     tmp(:, 7:8) = elem_cand(m);
%     tmp_matrix = [tmp_matrix; tmp];
% end
% new_IRS_sampleMatrix = [new_IRS_sampleMatrix; tmp_matrix];
% lab_endIdx = [lab_endIdx size(new_IRS_sampleMatrix,1)];

% lab2 -- 水平距离
angle = (0: 360/num_irs: 359)';
% radius_cand = config.horizon_dist_vec;
radius_cand = 190; % comp
tmp_matrix = [];
for m = 1:length(radius_cand)
    irs_x = radius_cand(m) .* cosd(angle);
    irs_y = radius_cand(m) .* sind(angle);

    tmp = base_IRS_sampleMatrix;
    tmp(:, IRS_x_id) = irs_x;
    tmp(:, IRS_y_id) = irs_y;
    tmp = sortrows(tmp, [IRS_x_id IRS_y_id], {'ascend', 'ascend'});
    tmp_matrix = [tmp_matrix; tmp];
end
new_IRS_sampleMatrix = [new_IRS_sampleMatrix; tmp_matrix];
lab_endIdx = [lab_endIdx size(new_IRS_sampleMatrix,1)];

% % lab3 -- 高度
% height_cand = config.height_vec;
% tmp_matrix = [];
% for m = 1:length(height_cand)
%     tmp = base_IRS_sampleMatrix;
%     tmp(:, 3) = height_cand(m);
%     tmp_matrix = [tmp_matrix; tmp];
% end
% new_IRS_sampleMatrix = [new_IRS_sampleMatrix; tmp_matrix];
% lab_endIdx = [lab_endIdx size(new_IRS_sampleMatrix, 1)];

% % lab4 -- 朝向
% % 绕z轴旋转，偏斜面板
% cand_angle = config.rotate_vec;
% rot_z_cand = zeros(num_irs, length(cand_angle));
% for ang = 1:length(cand_angle)
%     rot_z_cand(:,ang) = mod((0:num_irs-1)' * (360/num_irs) + cand_angle(ang), 360);
% end
% rot_z_cand(rot_z_cand > 180) = rot_z_cand(rot_z_cand > 180) - 360;
% tmp_matrix = [];
% for ang = 1:size(rot_z_cand,2)
%     tmp = base_IRS_sampleMatrix;
%     tmp(:, 4) = rot_z_cand(:,ang);
%     tmp_matrix = [tmp_matrix; tmp];
% end
% new_IRS_sampleMatrix = [new_IRS_sampleMatrix; tmp_matrix];
% lab_endIdx = [lab_endIdx size(new_IRS_sampleMatrix, 1)];

% % lab5 -- 方向图
% Gi_cand = config.Gi_vec;
% tmp_matrix = [];
% for m = 1:length(Gi_cand)
%     tmp = base_IRS_sampleMatrix;
%     tmp(:, 9) = Gi_cand(m);
%     tmp_matrix = [tmp_matrix; tmp];
% end
% new_IRS_sampleMatrix = [new_IRS_sampleMatrix;tmp_matrix];
% lab_endIdx = [lab_endIdx size(new_IRS_sampleMatrix, 1)];

% build AIRS_set
AIRS_set = cell(num_irs, 1);
AIRS.interval_AIRS_Y = General.wavelength_center/2; % element space
AIRS.interval_AIRS_Z = General.wavelength_center/2;
for irs_sam = 1:size(new_IRS_sampleMatrix, 1)
    AIRS.x = new_IRS_sampleMatrix(irs_sam, 1);
    AIRS.y = new_IRS_sampleMatrix(irs_sam, 2);
    AIRS.z = new_IRS_sampleMatrix(irs_sam, 3);
    AIRS.zrot = new_IRS_sampleMatrix(irs_sam, 4);
    AIRS.yrot = new_IRS_sampleMatrix(irs_sam, 5);
    AIRS.xrot = new_IRS_sampleMatrix(irs_sam, 6);
    RotateMatrix = calRotateMatrix(1, AIRS.zrot, AIRS.yrot, AIRS.xrot);
    AIRS.boresight = RotateMatrix{1} * [1,0,0]'; 

    AIRS.M_Y = new_IRS_sampleMatrix(irs_sam, 7);
    AIRS.M_Z = new_IRS_sampleMatrix(irs_sam, 8);
    AIRS.M = AIRS.M_Y * AIRS.M_Z;
    AIRS.element_y_indices = 1-(AIRS.M_Y+1)/2:(AIRS.M_Y-1)/2;
    AIRS.element_z_indices = 1-(AIRS.M_Z+1)/2:(AIRS.M_Z-1)/2;

    AIRS.antenna_gain = new_IRS_sampleMatrix(irs_sam, 9);
    AIRS.q = AIRS.antenna_gain / 2 - 1;
    AIRS.amplify_power = 10^(new_IRS_sampleMatrix(irs_sam, 10)/10)/1000;

    %%% dynamic noisePower
    dynamicNoise_density = 10^(new_IRS_sampleMatrix(irs_sam, 11)/10)/1000;
    AIRS.dynamic_noise_power_per_sc = @(sc_interval)dynamicNoise_density * sc_interval;

    AIRS_set{irs_sam} = AIRS;
end

%% UE

% save pre random seed
pre_rand_status = rng;

% use config seed
rng(config.seed);

UE_x_id = 1;
UE_y_id = 2;

num_UE_vec = config.num_UE_vec;
UE_sampleMatrix = cell(length(num_UE_vec), 1);
for idx = 1:length(num_UE_vec)
    num_ue = num_UE_vec(idx);
    ue_x = rand(num_ue, 1) * 2*config.init_UE_radius - config.init_UE_radius;
    ue_y = rand(num_ue, 1) * 2*config.init_UE_radius - config.init_UE_radius;
    coord_2D = [ue_x ue_y];
    coord_2D = sortrows(coord_2D, [UE_x_id UE_y_id], {'ascend', 'ascend'});
    UE_sampleMatrix{idx} = coord_2D;
end
UE.z = 1.5;
UE.Antennas = 1;

% recover random seed
rng(pre_rand_status);

%% save data

save(input_file, "General", "General_sampleMatrix", "new_IRS_sampleMatrix", "UE_sampleMatrix", ...
    "lab_endIdx", "BS", "AIRS_set", "UE");

end

% % 面朝天空
% % 高度为10
% height = 10;
% rot_y_cand = -90;
% tmp_matrix = [];
% for m = 1:length(rot_y_cand)
%     tmp = base_IRS_sampleMatrix;
%     tmp(:, 5) = rot_y_cand;
%     tmp(:, 4) = 0;
%     tmp(:, 3) = height;
%     tmp_matrix = [tmp_matrix ; tmp];
% end
% new_IRS_sampleMatrix = [new_IRS_sampleMatrix;tmp_matrix];
% 
% % 高度为5
% height = 5;
% rot_y_cand = -90;
% tmp_matrix = [];
% for m = 1:length(rot_y_cand)
%     tmp = base_IRS_sampleMatrix;
%     tmp(:, 5) = rot_y_cand;
%     tmp(:, 4) = 0;
%     tmp(:, 3) = height;
%     tmp_matrix = [tmp_matrix ; tmp];
% end
% new_IRS_sampleMatrix = [new_IRS_sampleMatrix;tmp_matrix];
