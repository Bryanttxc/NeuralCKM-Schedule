% created by Bryanttxc, FUNLab

function [AIRS, PathLoss, Power] = gene_channel_para(BS, AIRS, PathLoss, Power, Direction, Distance)
%GENE_CHANNEL_PARA generate S-V channel model parameters

% handle function
funcERP = @(theta, phi) AIRS.antenna_gain.*(sind(theta) .* cosd(phi)).^AIRS.q; % radiation pattern

%------------------ Large-scale Pathloss ------------------%
PathLoss.PL_BStoUE = PathLoss.avg_power_gain_ref .* Distance.dist_BStoUE_3D.^(-PathLoss.const_BStoUE);
PathLoss.PL_BStoAIRS = PathLoss.avg_power_gain_ref .* Distance.dist_BStoAIRS_3D.^(-PathLoss.const_BStoAIRS);
PathLoss.PL_AIRStoUE = PathLoss.avg_power_gain_ref .* Distance.dist_AIRStoUE_3D.^(-PathLoss.const_AIRStoUE);

%------------------ Radiation Pattern ------------------%
Power.ERP_BStoAIRS = funcERP(Direction.theta_AIRStoBS_local,Direction.phi_AIRStoBS_local);
if ~isfield(Direction, 'theta_AIRStoUE_path_local')
    Power.ERP_AIRStoUE = funcERP(Direction.theta_AIRStoUE_local,Direction.phi_AIRStoUE_local);
else
    Power.ERP_AIRStoUE = funcERP(Direction.theta_AIRStoUE_path_local,Direction.phi_AIRStoUE_path_local);
end

%------------------ Amplification factor ------------------%
AIRS.amplify_factor = sqrt( AIRS.amplify_power ./ ...
        (BS.trans_power * PathLoss.PL_BStoAIRS .* Power.ERP_BStoAIRS * AIRS.M + Power.dynamicNoise_power * AIRS.M) );

end
