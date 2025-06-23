% created by Bryanttxc, FUNLab

function [Direction] = gen_direction(BS, AIRS, UE, Distance, RotateMatrix)
%GENE_DIRECTION generate wave direction

% Coordinate system centered on the IRS
local_x_axis = (RotateMatrix{1} * [1,0,0]')';
local_y_axis = (RotateMatrix{1} * [0,1,0]')';
local_z_axis = (RotateMatrix{1} * [0,0,1]')';

phi_AIRStoBS_local             = acosd( local_x_axis(1:2) * (BS.position(1:2) - AIRS.position(1:2))' / Distance.dist_BStoAIRS_2D);
symbol_phi                     = acosd( local_y_axis(1:2) * (BS.position(1:2) - AIRS.position(1:2))' / Distance.dist_BStoAIRS_2D) > 90;
Direction.phi_AIRStoBS_local   = (-1)^symbol_phi * phi_AIRStoBS_local;
Direction.theta_AIRStoBS_local = acosd( local_z_axis*(BS.position - AIRS.position)' / Distance.dist_BStoAIRS_3D);

phi_AIRStoUE_local             = acosd( local_x_axis(1:2) * (UE.position(1:2) - AIRS.position(1:2))' / Distance.dist_AIRStoUE_2D);
symbol_phi                     = acosd( local_y_axis(1:2) * (UE.position(1:2) - AIRS.position(1:2))' / Distance.dist_AIRStoUE_2D) > 90;
Direction.phi_AIRStoUE_local   = (-1)^symbol_phi * phi_AIRStoUE_local;
Direction.theta_AIRStoUE_local = acosd( local_z_axis*(UE.position - AIRS.position)' / Distance.dist_AIRStoUE_3D);

% direction vector
Direction.direct_AIRStoBS_local = [sind(Direction.theta_AIRStoBS_local)*cosd(Direction.phi_AIRStoBS_local), sind(Direction.theta_AIRStoBS_local)*sind(Direction.phi_AIRStoBS_local), cosd(Direction.theta_AIRStoBS_local)];
Direction.direct_AIRStoUE_local = [sind(Direction.theta_AIRStoUE_local)*cosd(Direction.phi_AIRStoUE_local), sind(Direction.theta_AIRStoUE_local)*sind(Direction.phi_AIRStoUE_local), cosd(Direction.theta_AIRStoUE_local)];

% %%% debug
% Direction.theta_AIRStoBS_global = acosd( local_z_axis*(BS.position - AIRS.position)' / Distance.dist_BStoAIRS_3D );
% Direction.theta_AIRStoUE_global = acosd( local_z_axis*(UE.position - AIRS.position)' / Distance.dist_AIRStoUE_3D );
% Direction.phi_AIRStoBS_global = acosd( [1,0]*(BS.position(1:2) - AIRS.position(1:2))' / Distance.dist_BStoAIRS_2D );
% Direction.phi_AIRStoUE_global = acosd( [1,0]*(UE.position(1:2) - AIRS.position(1:2))' / Distance.dist_AIRStoUE_2D );
% Direction.direct_AIRStoBS_global = [sind(Direction.theta_AIRStoBS_global)*cosd(Direction.phi_AIRStoBS_global), sind(Direction.theta_AIRStoBS_global)*sind(Direction.phi_AIRStoBS_global), cosd(Direction.theta_AIRStoBS_global)];
% Direction.direct_AIRStoUE_global = [sind(Direction.theta_AIRStoUE_global)*cosd(Direction.phi_AIRStoUE_global), sind(Direction.theta_AIRStoUE_global)*sind(Direction.phi_AIRStoUE_global), cosd(Direction.theta_AIRStoUE_global)];

end
