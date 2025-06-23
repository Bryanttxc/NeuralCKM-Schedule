% created by Bryanttxc, FUNLab

function [BS, AIRS, UE, Distance, Direction] = gen_layout(BS, AIRS, UE, RotateMatrix)
%GENE_LAYOUT generate layout

% generate the position
[BS, AIRS, UE] = gen_position(BS, AIRS, UE);

% generate the distance
Distance = gen_distance(BS, AIRS, UE);

% generate the direction
Direction = gen_direction(BS, AIRS, UE, Distance, RotateMatrix);

end
