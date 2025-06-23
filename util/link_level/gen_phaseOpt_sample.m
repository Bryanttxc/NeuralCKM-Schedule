% created by Bryanttxc, FUNLab

function [BS, AIRS1, AIRS2, UE, Power, General] = gen_phaseOpt_sample(config)
%GEN_PHASEOPT_SAMPLE generate samples for phaseOpt_compare.m 

%% General parameters
% band
% 4G standard: center_freq->2GHz, BW->20MHz
% 5G standard: center_freq->3.5GHz, BW->100MHz

General.light_speed = 299792458; % m/s, consistent with QuaDRiGa
General.freq_center = config.freq_center; % GHz
General.wavelength_center = General.light_speed / General.freq_center; % m

General.bandwidth = config.bandwidth; % GHz
General.sc_interval = config.sc_interval; % KHz
General.num_sc = floor(General.bandwidth / General.sc_interval); % num of subcarriers

General.sc_perRB = 12; % num of subcarriers per RB
General.num_RB = floor(General.bandwidth / (General.sc_interval * General.sc_perRB)); % num of RBs, close to num_sc/sc_perRB

freq_index = 0:General.num_sc-1;
General.freq_set = General.freq_center + ...
                General.bandwidth ./ General.num_sc * (freq_index-(General.num_sc-1)/2); % cited by [1] not centeralized

%% BS/UE parameters

% num of z-axis position and antennas
BS.z = 25; % m
UE.z = 1.5;

BS.Antennas = 1; % single isotropic
UE.Antennas = 1;

%% AIRS parameters

% ------------ AIRS 1 ------------ %

AIRS1.num = config.AIRS_num;
AIRS1.z = config.AIRS_height(1); % m

% power
AIRS1.antenna_gain = config.AIRS_max_ERP(1);
AIRS1.q = AIRS1.antenna_gain / 2 - 1;

amplify_power_dBm = config.amplify_power_dBm; % same as transmit power of BS
AIRS1.amplify_power = 10^((amplify_power_dBm)./10)/1000; % dBm -> W

Nv_density_dBmPerHz = config.AIRS_Nv_density(1); % dBm/Hz
AIRS1.dynamicNoise_power_density = 10^(Nv_density_dBmPerHz/10)/1000; % W/Hz

% rotation
AIRS1.zrot = config.AIRS_zrot(1);
AIRS1.yrot = config.AIRS_yrot(1);
AIRS1.xrot = config.AIRS_xrot(1);
RotateMatrix = calRotateMatrix(1, AIRS1.zrot, AIRS1.yrot, AIRS1.xrot);
AIRS1.boresight = RotateMatrix{1} * [1, 0, 0]';

% interval
AIRS1.interval_AIRS_Y = General.wavelength_center/2;
AIRS1.interval_AIRS_Z = General.wavelength_center/2;

% element
AIRS1.M_Y = config.AIRS_M_Y;
AIRS1.M_Z = config.AIRS_M_Z;
AIRS1.M = AIRS1.M_Y .* AIRS1.M_Z;
AIRS1.element_y_indices = 1-(AIRS1.M_Y+1)/2:(AIRS1.M_Y-1)/2; % consistent with QuaDRiGa
AIRS1.element_z_indices = 1-(AIRS1.M_Z+1)/2:(AIRS1.M_Z-1)/2;

% ------------ AIRS 2 ------------ %
AIRS2.num = config.AIRS_num;
AIRS2.z = config.AIRS_height(2); % m

% power
AIRS2.antenna_gain = config.AIRS_max_ERP(2);
AIRS2.q = AIRS2.antenna_gain / 2 - 1;

amplify_power_dBm = config.amplify_power_dBm; % same as transmit power of BS
AIRS2.amplify_power = 10^((amplify_power_dBm)./10)/1000; % dBm -> W

Nv_density_dBmPerHz = config.AIRS_Nv_density(2); % dBm/Hz
AIRS2.dynamicNoise_power_density = 10^(Nv_density_dBmPerHz/10)/1000; % W/Hz

% rotation
AIRS2.zrot = config.AIRS_zrot(2);
AIRS2.yrot = config.AIRS_yrot(2);
AIRS2.xrot = config.AIRS_xrot(2);
RotateMatrix = calRotateMatrix(1, AIRS2.zrot, AIRS2.yrot, AIRS2.xrot);
AIRS2.boresight = RotateMatrix{1} * [1, 0, 0]';

% interval
AIRS2.interval_AIRS_Y = General.wavelength_center/2;
AIRS2.interval_AIRS_Z = General.wavelength_center/2;

% element
AIRS2.M_Y = config.AIRS_M_Y;
AIRS2.M_Z = config.AIRS_M_Z;
AIRS2.M = AIRS2.M_Y .* AIRS2.M_Z;
AIRS2.element_y_indices = 1-(AIRS2.M_Y+1)/2:(AIRS2.M_Y-1)/2; % consistent with QuaDRiGa
AIRS2.element_z_indices = 1-(AIRS2.M_Z+1)/2:(AIRS2.M_Z-1)/2;

%% Power parameters

% noise power
noise_density_dBmPerHz = -174; % dBm/Hz
noise_figure = 6; % dB
noise_density = 10.^((noise_density_dBmPerHz + noise_figure)./10)/1000; % dBm/Hz->W/Hz
Power.noise_power = noise_density * General.sc_interval; % W

% trans power
trans_power_dBm = config.trans_power_dBm; % dBm
BS.trans_power = 10^((trans_power_dBm)./10)/1000; % dBm -> W
Power.trans_power_per_sc = BS.trans_power / General.num_sc; % per subcarrier

% dynamic noise power
Power.AIRS1_dynamicNoise_power = AIRS1.dynamicNoise_power_density * General.sc_interval; % W
Power.AIRS2_dynamicNoise_power = AIRS2.dynamicNoise_power_density * General.sc_interval; % W

end
