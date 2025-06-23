%{
  Authors:  Ilya Burtakov, Andrey Tyarin

  Wireless Networks Lab, Institute for Information Transmission Problems 
  of the Russian Academy of Sciences and 
  Telecommunication Systems Lab, HSE University.

  testbeds@wireless.iitp.ru

  You may get the latest version of the QRIS platform at
        https://wireless.iitp.ru/ris-quadriga/

  If you are using QRIS, please kindly cite the paper 

  I. Burtakov, A. Kureev, A. Tyarin and E. Khorov, 
  "QRIS: A QuaDRiGa-Based Simulation Platform for Reconfigurable Intelligent Surfaces," 
  in IEEE Access, vol. 11, pp. 90670-90682, 2023, doi: 10.1109/ACCESS.2023.3306954.     

  https://ieeexplore.ieee.org/document/10225307
%}

function [effect_b, all_Nsub] = gene_quadriga_channel(smlt_para, BS, UE, AIRS_set, num_sc)

addpath(fullfile('channel', 'qris-tidy', 'class'));

%% generate BS, UE, AIRS Object

wavelength = smlt_para.wavelength;
center_frequency = smlt_para.center_frequency;

% Create a linear polarized antenna array of half-wave-dipole antennas for Tx
Tx = class_Tx_linear([BS.x, BS.y, BS.z], BS.Antennas, wavelength, 'omni', center_frequency); 

% Create a linear polarized antenna array of half-wave-dipole antennas for Rx
Rx = class_Rx_linear([UE.x, UE.y, UE.z], UE.Antennas, wavelength, 'omni', center_frequency);

% multi-AIRS (P.S. time-consuming: generate rotate pattern)
AIRS_COS = cell(length(AIRS_set), 1);
for irs_idx = 1:length(AIRS_set)
    AIRS = AIRS_set{irs_idx};
    AIRS_COS{irs_idx} = class_RIS_singlepol([AIRS.x, AIRS.y, AIRS.z], AIRS.M_Y, AIRS.M_Z, ...
                                             wavelength, center_frequency, ...
                                             AIRS.antenna_gain, AIRS.q, ...
                                             AIRS.xrot, AIRS.yrot, AIRS.zrot); 
end

%% Create QuaDRiGa layout

scenario_name_LOS = '3GPP_38.901_UMa_LOS';
scenario_name_NLOS = '3GPP_38.901_UMa_NLOS';
l = func_get_QuaDRiGa_layout(smlt_para, {Tx}, {Rx}, AIRS_COS, scenario_name_LOS, scenario_name_NLOS); % QRIS author created

%% Visualize scenario and print figure

% h = l.visualize;
% set(h,'PaperSize',[23.5 17]);

%% Creating index set of subcarriers including whole and UE

all_Nsub = (-(num_sc-1)/2 : (num_sc-1)/2) / num_sc;
% UE_Nsub = (-(num_sc_for_ue-1)/2 : (num_sc_for_ue-1)/2) / num_sc_for_ue;

%% Generate builders

b = l.init_builder;
sic = size( b );
effect_b = qd_builder;
cnt = 1;
for ib = 1 : numel(b)
    [ i1,i2 ] = qf.qind2sub( sic, ib ); % 数组的1-D index -> 矩阵的X-D index
    if ~isempty(b(i1,i2).tx_position) && ~isempty(b(i1,i2).rx_positions)
        effect_b(1,cnt) = b(i1,i2);
        effect_b(1,cnt).scenpar_nocheck = effect_b(1,cnt).scenpar;
        cnt = cnt + 1;
    end
end

end
