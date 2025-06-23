% created by Bryanttxc, FUNLab

function [Distance] = gen_distance(BS, AIRS, UE)
%GENE_DISTANCE generate distance

% handle function
funcDistance3D = @(a_3D,b_3D) sqrt(sum((a_3D-b_3D).^2, 2)); % 3D distance

Distance.dist_BStoAIRS_2D = funcDistance3D(BS.position(1:2), AIRS.position(1:2));
Distance.dist_AIRStoUE_2D = funcDistance3D(AIRS.position(1:2), UE.position(1:2));

Distance.dist_BStoUE_3D = funcDistance3D(BS.position, UE.position);
Distance.dist_BStoAIRS_3D = funcDistance3D(BS.position, AIRS.position);
Distance.dist_AIRStoUE_3D = funcDistance3D(AIRS.position, UE.position);

end
