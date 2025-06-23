% created by Bryanttxc, FUNLab

function [obj] = calOBJ(MCCM, phi)
% CALOBJ calcalute objective value

num_sc = size(MCCM,3);
obj = 0;
for sc = 1:num_sc
    obj = obj + real(phi' * MCCM(:,:,sc) * phi);
end

end