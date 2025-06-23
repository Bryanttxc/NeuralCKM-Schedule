% created by Bryanttxc, FUNLab

function [G_n, G_n_u_l] = calRadiationPattern(w_b, w_I_n, w_u_l, b_n, N, d_n_3D, d_n_u_l_3D, G_i, q, Lr)
%CALRADIATIONPATTERN	calculate the radiation pattern of IRS
% w_b, w_I_n, w_u_l: locations of BS, IRS n, multipath of UE u
% b_n: boresight of AIRS n
% N: num of AIRSs
% d_n_3D, d_n_u_l_3D: distance of BS-AIRS n, AIRS n-UE u links
% G_i, q: radiation coefficients
% Lr: num of multipath for UE

funcRadiaPattern = @(omega)G_i.*(cosd(omega)).^q;
omega_n = acosd(sum(b_n.*(w_b-w_I_n), 2) ./ d_n_3D); % BS-AIRS n link
omega_n_u_l = zeros(Lr,N);
for idxIRS = 1:N
    omega_n_u_l(:,idxIRS) = acosd(sum(b_n(idxIRS,:).*(w_u_l-w_I_n(idxIRS,:)), 2) ./ d_n_u_l_3D(:,idxIRS)); % uniform distributed in (0,90)
end

idx = omega_n <= 90;
G_n = funcRadiaPattern(omega_n) .* idx;

idx = omega_n_u_l <= 90;
G_n_u_l = funcRadiaPattern(omega_n_u_l) .* idx;

end