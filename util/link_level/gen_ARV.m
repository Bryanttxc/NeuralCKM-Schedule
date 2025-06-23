% created by Bryanttxc, FUNLab

function [ARV_incident, ARV_reflect, ARV_y, ARV_z] = gen_ARV(AIRS, Direction, light_speed, num_sc, freq_set, RotateMatrix)
%GENE_ARV generate array response vector

% handle function
funcARV = @(numAnt,freq,phase) 1./sqrt(numAnt) .* exp(1j*2*pi/light_speed.*freq.*phase); % array response vector

AIRS.element_space_y_arr = [zeros(1,AIRS.M_Y);
                            AIRS.element_y_indices.*AIRS.interval_AIRS_Y;  
                            zeros(1,AIRS.M_Y)]; % 3 x M_Y
AIRS.element_space_z_arr = [zeros(1,AIRS.M_Z); 
                            zeros(1,AIRS.M_Z); 
                            AIRS.element_z_indices.*AIRS.interval_AIRS_Z]; % 3 x M_Z

% AIRS -> BS
azimuth_phase_shift_AIRStoBS = (Direction.direct_AIRStoBS_local * AIRS.element_space_y_arr)'; % M_Y x 1
elevation_phase_shift_AIRStoBS = (Direction.direct_AIRStoBS_local * AIRS.element_space_z_arr)'; % M_Z x 1

% AIRS -> UE
azimuth_phase_shift_AIRStoUE = (Direction.direct_AIRStoUE_local * AIRS.element_space_y_arr)';
elevation_phase_shift_AIRStoUE = (Direction.direct_AIRStoUE_local * AIRS.element_space_z_arr)';


% %%% debug
% AIRS.element_space_y_arr_rotate = RotateMatrix{1} * [zeros(1,AIRS.M_Y);
%                                                      AIRS.element_y_indices.*AIRS.interval_AIRS_Y;  
%                                                      zeros(1,AIRS.M_Y)]; % 3 x M_Y
% AIRS.element_space_z_arr_rotate = RotateMatrix{1} * [zeros(1,AIRS.M_Z); 
%                                                      zeros(1,AIRS.M_Z); 
%                                                      AIRS.element_z_indices.*AIRS.interval_AIRS_Z]; % 3 x M_Z
% azimuth_pathlen_delta_BStoAIRS_rotate = (Direction.direct_AIRStoBS_global * AIRS.element_space_y_arr_rotate)'; % M_Y x 1
% elevation_pathlen_delta_BStoAIRS_rotate = (Direction.direct_AIRStoBS_global * AIRS.element_space_z_arr_rotate)'; % M_Z x 1
% 
% azimuth_pathlen_delta_AIRStoUE_rotate = (Direction.direct_AIRStoUE_global * AIRS.element_space_y_arr_rotate)';
% elevation_pathlen_delta_AIRStoUE_rotate = (Direction.direct_AIRStoUE_global * AIRS.element_space_z_arr_rotate)';    
% 
% % Logger
% temp = zeros(3, AIRS.M_Y * AIRS.M_Z);
% for m_z = 1:AIRS.M_Z
%     for m_y = 1:AIRS.M_Y
%         temp(:, m_y + (m_z - 1) * AIRS.M_Y) = AIRS.element_space_y_arr(:,m_y) + AIRS.element_space_z_arr(:,m_z);
%     end
% end
% Test.element_space_arr = temp;
% 
% Test.azimuth_pathlen_delta_AIRStoBS = azimuth_pathlen_delta_AIRStoBS;
% Test.elevation_pathlen_delta_AIRStoBS = elevation_pathlen_delta_AIRStoBS;
% Test.azimuth_pathlen_delta_AIRStoUE = azimuth_pathlen_delta_AIRStoUE;
% Test.elevation_pathlen_delta_AIRStoUE = elevation_pathlen_delta_AIRStoUE;
% 
% temp = zeros(3, AIRS.M_Y * AIRS.M_Z);
% for m_z = 1:AIRS.M_Z
%     for m_y = 1:AIRS.M_Y
%         temp(:, m_y + (m_z - 1) * AIRS.M_Y) = AIRS.element_space_y_arr_rotate(:,m_y) + AIRS.element_space_z_arr_rotate(:,m_z);
%     end
% end
% Test.element_space_arr_rotate = temp;
% 
% Test.azimuth_pathlen_delta_BStoAIRS_rotate = azimuth_pathlen_delta_BStoAIRS_rotate;
% Test.elevation_pathlen_delta_BStoAIRS_rotate = elevation_pathlen_delta_BStoAIRS_rotate;
% Test.azimuth_pathlen_delta_AIRStoUE_rotate = azimuth_pathlen_delta_AIRStoUE_rotate;
% Test.elevation_pathlen_delta_AIRStoUE_rotate = elevation_pathlen_delta_AIRStoUE_rotate;


%% ARV

ARV_incident = zeros(AIRS.M, num_sc);
ARV_reflect = zeros(AIRS.M, num_sc);
ARV_y = zeros(AIRS.M_Y, num_sc);
ARV_z = zeros(AIRS.M_Z, num_sc);

for sc = 1:num_sc

    ARV_incident(:,sc) = kron( funcARV(AIRS.M_Z,freq_set(sc),elevation_phase_shift_AIRStoBS), ...
                               funcARV(AIRS.M_Y,freq_set(sc),azimuth_phase_shift_AIRStoBS) );

    ARV_reflect(:,sc)  = kron( funcARV(AIRS.M_Z,freq_set(sc),elevation_phase_shift_AIRStoUE), ...
                               funcARV(AIRS.M_Y,freq_set(sc),azimuth_phase_shift_AIRStoUE) );

    ARV_y(:,sc) = funcARV(AIRS.M_Y, freq_set(sc), azimuth_phase_shift_AIRStoBS) .* ...
                    funcARV(AIRS.M_Y,freq_set(sc),azimuth_phase_shift_AIRStoUE);

    ARV_z(:,sc) = funcARV(AIRS.M_Z, freq_set(sc), elevation_phase_shift_AIRStoBS) .* ...
                    funcARV(AIRS.M_Z, freq_set(sc), elevation_phase_shift_AIRStoUE);

end

end
