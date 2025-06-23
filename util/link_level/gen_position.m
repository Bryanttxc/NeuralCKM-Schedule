% created by Bryanttxc, FUNLab

function [BS, AIRS, UE] = gen_position(BS, AIRS, UE)
%GENE_POSITION generate position

BS.position = cal_coordinate(BS.x, BS.y, BS.z);

AIRS.position = cal_coordinate(AIRS.x, AIRS.y, AIRS.z);

UE.position = cal_coordinate(UE.x, UE.y, UE.z);

end

function [position] = cal_coordinate(x, y, z)
%CALCOORDINATE   generate the 3D cartesian coordinate

position = [x, y, z];

end
