% created by Bryanttxc, FUNLab

function gen_link_level_data(config, input_file, num_IRS)
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
new_IRS_sampleMatrix = [irs_x irs_y irs_z rot_z rot_y rot_x M_Y M_Z G_i Pa Nv_density];

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

x_sam = -config.init_UE_radius : config.interval : config.init_UE_radius;
y_sam = x_sam;
UE_x = reshape(repmat(x_sam, length(y_sam), 1), [], 1);
UE_y = repmat(y_sam', length(x_sam), 1);
UE_sampleMatrix = [UE_x UE_y];

UE.z = 1.5;
UE.Antennas = 1;

%% save data

save(input_file, "General", "General_sampleMatrix", "new_IRS_sampleMatrix", ...
        "UE_sampleMatrix", "BS", "AIRS_set", "UE");

end