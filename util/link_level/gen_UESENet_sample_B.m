% created by Bryanttxc, FUNLab

function [sampleMatrix, BS, AIRS, UE, Power, General] = gen_UESENet_sample_B()

%% General Parameter
numSample = 1e4; % sample number

General.light_speed = 299792458; % light speed, consistent with quadriga
General.freq_center = 3.5e9; % central freq
General.wavelength_center = General.light_speed / General.freq_center; % m

bandwidth = [2;5;10;20;50;100] .* 1e6; % MHz
bandwidth_sample = randShuffle(bandwidth, numSample);

numerology = 1:4; % 5G NR
numerology_sample = randShuffle(numerology, numSample);
sc_interval = [15;30;60;120] .* 1e3; % KHz
sc_interval_sample = sc_interval(numerology_sample);

General.SCperRB = 12; % num of subcarriers per RB
num_RB_sample = floor(bandwidth_sample ./ (sc_interval_sample * General.SCperRB)); % num of RBs
rho_sample = zeros(numel(num_RB_sample), 1);
for sam = 1:numel(num_RB_sample)
    rho_set = (1:num_RB_sample(sam)) ./ num_RB_sample(sam);
    R = randperm(length(rho_set));
    rho_sample(sam) = rho_set(R(1));
end

cellRadius = 250; % 3GPP: intersite distance (ISD) or cell diameter 500m for UMa, 
                  % 200m for UMi.
General.num_fading = 1000; % num of fading realizations

%% Power
% noise power
noise_density_dBmPerHz = -174; % dBm/Hz
noise_figure = 6; % dB
noise_density = 10^((noise_density_dBmPerHz + noise_figure)/10)/1000; % dBm/Hz->W/Hz
Power.noise_power_per_sc_sample = noise_density .* sc_interval_sample; % W

% trans power
trans_power_dBm = 10:10:40; % 10dBm-40dBm
trans_power = 10.^(trans_power_dBm ./10)./1000; % dBm -> W
trans_power_sample = randShuffle(trans_power, numSample); % generate samples
Power.trans_power_per_sc_sample = trans_power_sample ./ (num_RB_sample .* General.SCperRB);
% ts_snr_sample = 10.*log10(ts_per_sc_sample ./ Power.noise_power_per_sc_sample); % dB

%% BS
BS.x = 0;
BS.y = 0;
BS.z = 25; % m

BS.Antennas = 1;

%% AIRS
AIRS.num = 1;
AIRS.interval_AIRS_Y = General.wavelength_center/2; % element space
AIRS.interval_AIRS_Z = General.wavelength_center/2;

% x, y, z
irs_x_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % m
irs_y_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % m
irs_z_sample = rand(numSample, 1) .* 34 + 1; % 1-35 m

% rotate angle
angleMax = 360;
zrot_sample = rand(numSample, 1) .* angleMax - angleMax/2; % rotate along z-axis
yrot_sample = rand(numSample, 1) .* angleMax - angleMax/2; % rotate along y-axis
% xrot_sample = rand(numSample, 1) .* angleMax - angleMax/2; % rotate along x-axis (y-z plane)

% % calibration for ensuring BS-AIRS LoS case (no used)
% direction_AIRStoBS = [zeros(numSample,1)-irs_x_sample zeros(numSample,1)-irs_y_sample];
% azimuth_angle_AIRStoBS = acosd( direction_AIRStoBS * [1,0]' ./ sqrt(sum(direction_AIRStoBS.^2, 2)) );
% isPositive = direction_AIRStoBS(:,2) < 0;
% azimuth_angle_AIRStoBS = (-1).^isPositive .* azimuth_angle_AIRStoBS;
% zrot_sample = zeros(numSample, 1);
% for sam = 1:numSample
%     % range: azimuth_angle_AIRStoBS(idx) - 90 to azimuth_angle_AIRStoBS(idx) + 90
%     angle = rand() * 180 + (azimuth_angle_AIRStoBS(sam) - 90);
%     if angle <= -180
%         angle = angle + 360;
%     elseif angle > 180
%         angle = angle - 360;
%     end
%     zrot_sample(sam) = angle;
% end

% element num
M_Y = 8:2:16;
M_Z = 8:2:16;

M_Y_sample = randShuffle(M_Y, numSample);
M_Z_sample = randShuffle(M_Z, numSample);

% element radiation pattern
q = 1:5;
q_sample = randShuffle(q, numSample);
G_i_sample = (q_sample + 1) .* 2;

% IRS amplification power
Pa_dBm = 0:10:30; % 0dBm-30dBm, refer to the folders: evidences_of_amplify_power
% Pa = 10.^(Pa_dBm ./10)./1000; % dBm -> W
Pa_dBm_sample = randShuffle(Pa_dBm, numSample); % generate samples

NvdBm_density_sample = rand(numSample, 1) .* 10 - 160; % -150~-160dBm/Hz, refer to the folders: evidences_of_amplify_power
Nv_density_sample = 10.^(NvdBm_density_sample ./10)./1000; % dBm/Hz -> W/Hz
dyn_noi_power_per_sc_dBm_sample = 10.*log10(Nv_density_sample .* sc_interval_sample .* 1e3);

%% UE
% x, y, z
ue_x_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % m
ue_y_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % m
UE.z = 1.5; % m, 3GPP rule

UE.Antennas = 1;

%% isBeamforming
% 0: beamforming
% 1: scattering
% 2: no IRS
isBeamform = [0;1;2];
isBeamform_sample = randShuffle(isBeamform, numSample);

% % calibration (no used)
% direction_AIRStoUE = [ue_x_sample-irs_x_sample ue_y_sample-irs_y_sample];
% for sam = 1:numSample
%     rotationMatrix = calRotateMatrix(1, zrot_sample(sam), yrot_sample(sam), xrot_sample(sam));
%     boresight = rotationMatrix{1} * [1,0,0]'; % 3 x 1
%     azimuth_angle_AIRStoUE = acosd(direction_AIRStoUE(sam,:) * boresight(1:2) ./ sqrt(sum(direction_AIRStoUE(sam,:).^2, 2)) );
%     if azimuth_angle_AIRStoUE > 90
%         isBeamform_sample(sam) = randi([1 2]);
%     end
% end
% beamform_num = length(find(isBeamform_sample == 0));
% scatter_num = length(find(isBeamform_sample == 1));
% noIRS_num = length(find(isBeamform_sample == 2));
% fprintf("beamform num: %d, scatter_num = %d, noIRS_num = %d\n", beamform_num, scatter_num, noIRS_num);

%% Sample Matrix
sampleMatrix = [];
sampleMatrix = [sampleMatrix trans_power_sample]; % 1
sampleMatrix = [sampleMatrix irs_x_sample irs_y_sample irs_z_sample]; % 2 3 4
sampleMatrix = [sampleMatrix zrot_sample yrot_sample]; % 5 6
sampleMatrix = [sampleMatrix M_Y_sample M_Y_sample]; % 7 8
sampleMatrix = [sampleMatrix G_i_sample]; % 9
sampleMatrix = [sampleMatrix Pa_dBm_sample dyn_noi_power_per_sc_dBm_sample]; % 10 11
sampleMatrix = [sampleMatrix ue_x_sample ue_y_sample]; % 12 13
sampleMatrix = [sampleMatrix rho_sample]; % 14
sampleMatrix = [sampleMatrix bandwidth_sample sc_interval_sample isBeamform_sample]; % 15 16 17

end