% created by Bryanttxc, FUNLab

function [Loc_irs] = calRotationIRS(Loc_IRS_center, N_H, N_V, d_irs_H, d_irs_V, alpha, beta, gamma, plane) 

%%%%%%%%%%%%
% cited by "3GPP TR 38.901 v14.2.0(2017-09) 7.1.3 Transformation from a LCS to a GCS"
% 旋转正方向遵从右手定则 (The positive direction of rotation obeys the right-hand rule)
%%%%%%%%%%%%

% Loc_irs_center dimension: 3 x 1
if size(Loc_IRS_center,1) ~= 3
    error("invalid dimension: Loc_irs_center");
end

% reference coordinate in GCS
Loc_irs_bar = zeros(3,N_H*N_V);

% choose the IRS plane
if strcmp(plane,'x-y') == 1
    Loc_irs_bar(1,:) = reshape( repmat((-N_H/2:N_H/2-1)*d_irs_H, N_V, 1), 1, []);
    Loc_irs_bar(2,:) = repmat( (-N_V/2:N_V/2-1)*d_irs_V, 1, N_H );
elseif strcmp(plane,'x-z') == 1
    Loc_irs_bar(1,:) = reshape( repmat((-N_H/2:N_H/2-1)*d_irs_H, N_V, 1), 1, []);
    Loc_irs_bar(3,:) = repmat( (-N_V/2:N_V/2-1)*d_irs_V, 1, N_H );
elseif strcmp(plane,'y-z') == 1
    Loc_irs_bar(2,:) = reshape( repmat((-N_H/2:N_H/2-1)*d_irs_H, N_V, 1), 1, []);
    Loc_irs_bar(3,:) = repmat( (-N_V/2:N_V/2-1)*d_irs_V, 1, N_H );
else
    error("input wrong!");
end

Rz = [cosd(alpha) -sind(alpha)  0;
      sind(alpha)  cosd(alpha)  0;       
          0           0         1]; % rotation matrix along z-axis

Ry = [cosd(beta)      0     sind(beta);
           0          1         0;       
      -sind(beta)     0     cosd(beta)]; % rotation matrix along y-axis

Rx = [ 1           0          0;
       0      cosd(gamma) -sind(gamma);      
       0      sind(gamma)  cosd(gamma)]; % rotation matrix along x-axis

Loc_irs = Loc_IRS_center + Rz * Ry * Rx * Loc_irs_bar;
Loc_irs = Loc_irs'; % dimension: N_H*N_V x 3

end