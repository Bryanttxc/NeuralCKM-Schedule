% created by Bryanttxc, FUNLab
% reference:
% [1] Chen Y, Chen D, Jiang T. Beam-squint mitigating in reconfigurable intelligent surface aided wideband mmWave communications[C]
% //2021 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2021: 1-6.

function [General, Power, PathLoss, BS, AIRS, UE] = oneBS_oneIRS_oneUE_scene(config)

%% general parameters

General.light_speed = config.light_speed; % m/s
General.freq_center = config.freq_center; % GHz
General.wavelength_center = General.light_speed ./ General.freq_center; % m

General.bandwidth = config.bandwidth; % MHz
General.num_sc = config.num_sc; % total num of subcarriers
General.sc_interval = General.bandwidth ./ General.num_sc;

freq_index = 1:General.num_sc;
General.freq_set = General.freq_center + ...
                        General.bandwidth ./ General.num_sc .* (freq_index-1-(General.num_sc-1)./2); % cited by [1]

General.num_path_BStoUE = config.num_path_BStoUE;
General.num_path_AIRStoUE = config.num_path_AIRStoUE;
General.max_delay = config.max_delay;

% handle function
General.funcARV = @(numAnt,freq,phase) 1/sqrt(numAnt).* exp(1j*2*pi/light_speed.*freq.*phase); % array response vector
General.funcDelay = @(f,tau)exp(-1j*2*pi*tau.*f); % delay term
General.funcDistance3D = @(a_3D,b_3D) sqrt(sum((a_3D-b_3D).^2, 2)); % 3D distance
General.funcERP = @(Gi, q, theta, phi) Gi.*(sind(theta) .* cosd(phi)).^q; % radiation pattern

General.M_Y_sets = config.M_Y_sets;
General.M_Z_sets = config.M_Z_sets;
General.num_sc_sets = config.num_sc_sets; % subcarriers

%% Power struct

% noise power
noise_density_dBmPerHz = config.noise_density_dBmPerHz; % dBm/Hz
noise_figure = config.noise_figure; % dB
Power.noise_density = 10.^((noise_density_dBmPerHz+noise_figure)./10)/1000; % dBm/Hz->W/Hz

% trans power
trans_power_dBm = config.trans_power_dBm; % dBm
BS.trans_power = 10^((trans_power_dBm)./10)/1000; % dBm -> W

Power.noise_power = Power.noise_density * General.sc_interval; % W
Power.trans_power_per_sc = BS.trans_power ./ General.num_sc;

%% BS/UE struct

% position
BS.z = config.BS_z; % m
UE.z = config.UE_z;

%% PathLoss struct

PathLoss.avg_power_gain_ref = (General.wavelength_center / (4*pi))^2;
PathLoss.const_BStoUE = config.const_PL_BStoUE;
PathLoss.const_BStoAIRS = config.const_PL_BStoAIRS;
PathLoss.const_AIRStoUE = config.const_PL_AIRStoUE;

%% AIRS struct

AIRS.num = config.num_AIRS;
AIRS.antenna_gain = config.AIRS_antenna_gain;
AIRS.q = AIRS.antenna_gain / 2 - 1;
AIRS.amplify_power = BS.trans_power; % mW

dynamicNoise_density_dBmPerHz = config.AIRS_dynamicNoise_density; % dBm/Hz
AIRS.dynamicNoise_power_density = 10^(dynamicNoise_density_dBmPerHz/10); % mW/Hz
Power.dynamicNoise_power = AIRS.dynamicNoise_power_density * General.sc_interval;

% position
AIRS.interval_AIRS_Y = General.wavelength_center/2;
AIRS.interval_AIRS_Z = General.wavelength_center/2;
AIRS.M_Y = config.AIRS_M_Y;
AIRS.M_Z = AIRS.M_Y;
AIRS.M = AIRS.M_Y .* AIRS.M_Z;
AIRS.z = config.AIRS_z; % height
AIRS.horizon_rot_angle = config.AIRS_horizon_rot_angle;
AIRS.boresight = [cosd(AIRS.horizon_rot_angle), sind(AIRS.horizon_rot_angle), 0]; % panel boresight
AIRS.element_y_indices = 0:AIRS.M_Y-1; % 1-M_Y/2:M_Y/2
AIRS.element_z_indices = 0:AIRS.M_Z-1; % 1-M_Z/2:M_Z/2
