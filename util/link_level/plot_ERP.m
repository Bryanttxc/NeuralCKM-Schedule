% created by Bryanttxc, FUNLab

function plot_ERP(zrot, yrot, xrot, Gi)
%PLOT_ERP visualize Element Radiation Pattern

root_path = fullfile(fileparts(fileparts(pwd)), '');

addpath(fullfile(root_path, 'channel', 'quadriga_src'));
addpath(fullfile(root_path, 'channel', 'qris-tidy', 'class'));
addpath(fullfile(root_path, 'channel', 'qris-tidy', 'func'));

% Gi = 6;
% zrot = -53.2423;
% yrot = -38.8556;
% xrot = 52.3911;

q = Gi/2 - 1;
Ain = sqrt(Gi); Bin = 0; Cin = q; Din = 0;
COS_UC_pol = qd_arrayant('Parametric', Ain, Bin, Cin, Din);
COS_UC_pol.rotate_pattern([xrot,yrot,zrot], 'xyz', [], 1);
COS_UC_pol.visualize(1);
set(gca, 'fontsize', 28);

end
