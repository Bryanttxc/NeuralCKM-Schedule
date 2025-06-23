% created by Bryanttxc, FUNLab

function [RotateMatrix] = calRotateMatrix(N, alpha, beta, gamma)
%CALROTATIONMATRIX   generate rotation matrix
% 
% %%% input %%%
% N: number of IRSs
% alpha, beta, gamma: rotation angle along z-, y-, x-axis
% %%% output %%%
%　RotateMatrix: rotation matrix

RotateMatrix = cell(N,1); % rotation matrix

for idxIRS = 1:N
    Rz = [cosd(alpha(idxIRS)) -sind(alpha(idxIRS)) 0;
          sind(alpha(idxIRS))  cosd(alpha(idxIRS)) 0;
                    0                     0        1];          % rotation matrix along z-axis

    Ry = [cosd(beta(idxIRS))     0     sind(beta(idxIRS));
                  0              1         0;       
         -sind(beta(idxIRS))     0     cosd(beta(idxIRS))];     % rotation matrix along y-axis

    Rx = [  1               0                   0;
            0      cosd(gamma(idxIRS)) -sind(gamma(idxIRS));      
            0      sind(gamma(idxIRS))  cosd(gamma(idxIRS))];   % rotation matrix along x-axis

    RotateMatrix{idxIRS} = Rz * Ry * Rx;
end

end