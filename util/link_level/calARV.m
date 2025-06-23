% created by Bryanttxc, FUNLab

function [ARV_incident,ARV_reflect,ARV_y,ARV_z] = calARV(AIRS, Direction, freq_set, RotateMatrix)
% CALARV    calculate the array response vector

light_speed = 3e8;
funcARV = @(numAnt,f,phase)1/sqrt(numAnt).* exp(1j*2*pi/light_speed.*f.*phase);

num_sc = length(freq_set);
num_path_AIRStoUE = size(Direction.theta_AIRStoUE_path_local,1);
num_angleTrial = size(Direction.theta_AIRStoUE_path_local,2);
AIRS.element_space_y_arr = RotateMatrix{1} * [zeros(1,AIRS.M_Y); 
                                              AIRS.element_y_indices.*AIRS.interval_AIRS_Y;  
                                              zeros(1,AIRS.M_Y)]; % 3 x M_Y
AIRS.element_space_z_arr = RotateMatrix{1} * [zeros(1,AIRS.M_Z); 
                                              zeros(1,AIRS.M_Z); 
                                              AIRS.element_z_indices.*AIRS.interval_AIRS_Z]; % 3 x M_Z

%% BS-AIRS link

azimuth_pathlen_delta_BStoAIRS   = Direction.direct_AIRStoBS_local * AIRS.element_space_y_arr; % 1 x M_Y
elevation_pathlen_delta_BStoAIRS = Direction.direct_AIRStoBS_local * AIRS.element_space_z_arr; % 1 x M_Z

Direction.direct_AIRStoUE_local = zeros(num_path_AIRStoUE,3,num_angleTrial);
for nat = 1:num_angleTrial
    Direction.direct_AIRStoUE_local(:,:,nat) = [
                                      sind(Direction.theta_AIRStoUE_path_local(:,nat)).*cosd(Direction.phi_AIRStoUE_path_local(:,nat)),...
                                      sind(Direction.theta_AIRStoUE_path_local(:,nat)).*sind(Direction.phi_AIRStoUE_path_local(:,nat)),...
                                      cosd(Direction.theta_AIRStoUE_path_local(:,nat))]; % path x 3
end

%% AIRS-UE link

ARV_incident = zeros(AIRS.M,num_sc);
ARV_reflect = zeros(AIRS.M,num_path_AIRStoUE,num_sc,num_angleTrial);
ARV_y = zeros(AIRS.M_Y,num_path_AIRStoUE,num_sc,num_angleTrial);
ARV_z = zeros(AIRS.M_Z,num_path_AIRStoUE,num_sc,num_angleTrial);
for sc = 1:num_sc
    ARV_incident(:,sc) = kron( funcARV(AIRS.M_Y,freq_set(sc),azimuth_pathlen_delta_BStoAIRS'),... 
                               funcARV(AIRS.M_Z,freq_set(sc),elevation_pathlen_delta_BStoAIRS') ); % M x 1
    for nat = 1:num_angleTrial
        azimuth_pathlen_delta_AIRStoUE = Direction.direct_AIRStoUE_local(:,:,nat) * AIRS.element_space_y_arr; % num_path x M_Y
        elevation_pathlen_delta_AIRStoUE = Direction.direct_AIRStoUE_local(:,:,nat) * AIRS.element_space_z_arr; % num_path x M_Z

        % accelarate by matrix operation, O(n^3) -> O(n^2)
        matrix = repmat(funcARV(AIRS.M_Y,freq_set(sc),azimuth_pathlen_delta_AIRStoUE), AIRS.M_Z, 1); % num_path x M_Y -> (M_Z x num_path) x M_Y
        arv_y = reshape(reshape(matrix,1,[]), num_path_AIRStoUE, []); % 1 x (num_path x M_Z x M_Y) -> num_path x M
        arv_z = repmat(funcARV(AIRS.M_Z,freq_set(sc),elevation_pathlen_delta_AIRStoUE), 1, AIRS.M_Y); % num_path x M
        ARV_reflect(:,:,sc,nat) = (arv_y .* arv_z).';

        % % benchmark version: O(n^3)
        % for np = 1:num_path_AIRStoUE
        %     ARV_R_TEMP(:,np,sc,nat) = kron( funcARV(AIRS.M_Y,freq_set(sc),azimuth_pathlen_delta_AIRStoUE(np,:)'),...
        %                                     funcARV(AIRS.M_Z,freq_set(sc),elevation_pathlen_delta_AIRStoUE(np,:)') ); % M x 1
        % end
        % 
        % % check two version whether is common
        % diff = ARV_R_TEMP(:,:,sc,nat) - ARV_reflect(:,:,sc,nat);
        % index = find(diff~=0);
        % if ~isempty(index)
        %     fprintf("ARV error!")
        % end

        % DC-SCA algorithm need
        for np = 1:num_path_AIRStoUE
           ARV_y(:,np,sc,nat) = funcARV(AIRS.M_Y,freq_set(sc),azimuth_pathlen_delta_BStoAIRS') .* funcARV(AIRS.M_Y,freq_set(sc),azimuth_pathlen_delta_AIRStoUE(np,:)') ;
           ARV_z(:,np,sc,nat) = funcARV(AIRS.M_Z,freq_set(sc),azimuth_pathlen_delta_BStoAIRS') .* funcARV(AIRS.M_Z,freq_set(sc),elevation_pathlen_delta_AIRStoUE(np,:)') ;
        end

    end % end angle
end % end sc

end