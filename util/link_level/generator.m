% created by Bryanttxc, FUNLab

function [General_sampleMatrix, IRS_sampleMatrix, UE_sampleMatrix, ...
          BS, AIRS_set, UE, Power, General] = generator()

%% General Parameter (no mistakes)
General.light_speed = 299792458; % exactly light speed, same as quadriga
General.freq_center = 3.5e9; % central frequency
General.wavelength_center = General.light_speed / General.freq_center; % m
General.num_sc_per_RB = 12; % num of subcarriers per RB

bandwidth = [2 5 10 20 50 100] .* 1e6; % MHz, 6 cases
numerology = (1:3)'; % 5G NR, 3 cases, suitable for sub-6GHz
sc_interval = [15;30;60] .* 1e3; % KHz
num_bw = length(bandwidth);
num_numero = length(numerology);

bandwidth_sample = reshape(repmat(bandwidth, num_numero, 1), [], 1);
numerology_sample = repmat(numerology, num_bw, 1);
sc_interval_sample = sc_interval(numerology_sample);

bandwidth_sample(16) = []; % delete invalid bw-sc_interv combination
sc_interval_sample(16) = [];

num_RB_sample = floor(bandwidth_sample ./ (sc_interval_sample .* General.num_sc_per_RB)); % num of RBs

General_sampleMatrix = zeros(sum(num_RB_sample), 5);
cur = 1;
for sam = 1:numel(num_RB_sample)
    rho_set = (1:num_RB_sample(sam))' ./ num_RB_sample(sam);
    tmp_bw_sample    = repmat(bandwidth_sample(sam), length(rho_set), 1);
    tmp_sc_sample    = repmat(sc_interval_sample(sam), length(rho_set), 1);
    tmp_numRB_sample = repmat(num_RB_sample(sam), length(rho_set), 1);
    tmp_numsc_sample = repmat(num_RB_sample(sam)*General.num_sc_per_RB, length(rho_set), 1);
    General_sampleMatrix(cur:cur+length(rho_set)-1,:) = [tmp_bw_sample tmp_sc_sample tmp_numRB_sample tmp_numsc_sample rho_set];
    cur = cur + length(rho_set);
end

cellRadius = 250; % 3GPP: intersite distance (ISD) or cell diameter 500m for UMa, 
numSample = 1e4; % sample number

%% Power (no mistakes)
% noise density(fixed)
noise_density_dBmPerHz = -174; % dBm/Hz
noise_figure = 6; % dB
Power.noise_density = 10^((noise_density_dBmPerHz + noise_figure)/10)/1000; % dBm/Hz->W/Hz

% trans power
Power.trans_power_dBm = 10:10:40; % 10dBm-40dBm

%% BS (no mistakes)
BS.x = 0;
BS.y = 0;
BS.z = 25; % m

BS.Antennas = 1;

%% AIRS (no mistakes)

% x, y, z
irs_x_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % -250-250 m
irs_y_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % -250-250 m
irs_z_sample = rand(numSample, 1) .* 34 + 1; % 1-35 m

% rotate angle, quadriga rotate angular range:-180~180
angleMax = 360;
zrot_sample = rand(numSample, 1) .* angleMax - angleMax/2; % rotate along z-axis
yrot_sample = rand(numSample, 1) .* angleMax - angleMax/2; % rotate along y-axis
xrot_sample = rand(numSample, 1) .* angleMax - angleMax/2; % rotate along x-axis

% element num (assume M_Y==M_Z)
M_Y = 8:2:16;
% M_Z = 8:2:16;
M_Y_sample = randShuffle(M_Y, numSample);
% M_Z_sample = randShuffle(M_Z, numSample);

% element radiation pattern
q = 1:5;
q_sample = randShuffle(q, numSample);
G_i_sample = (q_sample + 1) .* 2;

% IRS amplification power
Pa_dBm = 0:10:30; % 0dBm-30dBm, refer to the folders: evidences_of_amplify_power
Pa_dBm_sample = randShuffle(Pa_dBm, numSample); % generate samples

% dynamic noise power
NvdBm_density_sample = rand(numSample, 1) .* 10 - 160; % -150~-160dBm/Hz, refer to the folders: evidences_of_amplify_power

% the whole IRS sample matrix
IRS_sampleMatrix = [irs_x_sample, irs_y_sample, irs_z_sample, ... % 1 2 3
                     zrot_sample, yrot_sample, xrot_sample,... % 4 5 6
                     M_Y_sample, M_Y_sample, ... % 7 8
                     G_i_sample, Pa_dBm_sample, NvdBm_density_sample]; % 9 10 11

% create AIRS struct array
AIRS_set = cell(numSample, 1);
AIRS.interval_AIRS_Y = General.wavelength_center/2; % element space
AIRS.interval_AIRS_Z = General.wavelength_center/2;
for irs_sam = 1:numSample
    AIRS.x = IRS_sampleMatrix(irs_sam, 1);
    AIRS.y = IRS_sampleMatrix(irs_sam, 2);
    AIRS.z = IRS_sampleMatrix(irs_sam, 3);
    AIRS.zrot = IRS_sampleMatrix(irs_sam, 4);
    AIRS.yrot = IRS_sampleMatrix(irs_sam, 5);
    AIRS.xrot = IRS_sampleMatrix(irs_sam, 6);
    RotateMatrix = calRotateMatrix(1, AIRS.zrot, AIRS.yrot, AIRS.xrot);
    AIRS.boresight = RotateMatrix{1} * [1,0,0]'; 
    
    AIRS.M_Y = IRS_sampleMatrix(irs_sam, 7);
    AIRS.M_Z = IRS_sampleMatrix(irs_sam, 8);
    AIRS.M = AIRS.M_Y * AIRS.M_Z;
    AIRS.element_y_indices = 1-(AIRS.M_Y+1)/2:(AIRS.M_Y-1)/2;
    AIRS.element_z_indices = 1-(AIRS.M_Z+1)/2:(AIRS.M_Z-1)/2;
    
    AIRS.antenna_gain = IRS_sampleMatrix(irs_sam, 9);
    AIRS.q = AIRS.antenna_gain / 2 - 1;
    AIRS.amplify_power = 10^(IRS_sampleMatrix(irs_sam, 10)/10)/1000;
    
    %%% dynamic noisePower
    dynamicNoise_density = 10^(IRS_sampleMatrix(irs_sam, 11)/10)/1000;
    AIRS.dynamic_noise_power_per_sc = @(sc_interval)dynamicNoise_density * sc_interval;
    
    AIRS_set{irs_sam} = AIRS;
end

%% UE (no mistakes)
% x, y, z
ue_x_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % -250-250 m
ue_y_sample = rand(numSample, 1) .* 2 * cellRadius - cellRadius; % -250-250 m
UE.z = 1.5; % m, 3GPP rule
UE.Antennas = 1;

UE_sampleMatrix = [ue_x_sample ue_y_sample];

end
