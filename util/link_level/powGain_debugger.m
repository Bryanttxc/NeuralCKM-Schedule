% created by Bryanttxc, FUNLab

function powGain_debugger(CCM, ARV_y, ARV_z, M_Y, M_Z, RadiaPattern_BStoAIRS, RadiaPattern_AIRStoUE)

M = M_Y * M_Z;
num_sc = size(CCM, 3);
freq_center_index = round(num_sc/2);
powGain_dB_benchmark_1 = zeros(1,num_sc);
powGain_dB_benchmark_2 = zeros(1,num_sc);
powGain_dB_benchmark_3 = zeros(1,num_sc);

% ---------------- benchmark 1 (Jiang Zhenyu version)---------------- %
% abs(ARV_y) = 1 / M_Y, abs(ARV_z) = 1 / M_Z
phi_fc_subopt_y = ARV_y(:,freq_center_index);
phi_fc_subopt_z = ARV_z(:,freq_center_index);
W_y_opt = conj(phi_fc_subopt_y) * phi_fc_subopt_y.';
W_z_opt = conj(phi_fc_subopt_z) * phi_fc_subopt_z.';

sumition_y = zeros(1,num_sc);
sumition_z = zeros(1,num_sc);
for sc = 1:num_sc
    covariance_iter = conj(ARV_y(:,sc)) * ARV_y(:,sc).';
    sumition_y(sc) = real(trace(covariance_iter * W_y_opt));
    covariance_other = conj(ARV_z(:,sc)) * ARV_z(:,sc).';
    sumition_z(sc) = real(trace(covariance_other * W_z_opt));
end
powGain_dB_benchmark_1 = 10*log10( (M^2) * sumition_y .* sumition_z .* RadiaPattern_AIRStoUE .* RadiaPattern_BStoAIRS);

% ---------------- benchmark 2 ---------------- % 
% CCM --> 1 / M
% abs(ARV_incident) = 1 / sqrt(M), abs(ARV_reflect) = 1 / sqrt(M)
W_opt = kron(W_y_opt,W_z_opt);
for sc = 1:num_sc
    powGain_dB_benchmark_2(sc) = 10*log10((M^2) * real(trace(CCM(:,:,sc) * W_opt)));
end

% ---------------- benchmark 3 ---------------- % 
% abs(w_y_opt) = 1, abs(w_z_opt) = 1
[U_y,S_y,~] = svd(W_y_opt);
[U_z,S_z,~] = svd(W_z_opt);
w_y_opt = exp(1j*angle(U_y(:,1)));
w_z_opt = exp(1j*angle(U_z(:,1)));
w_opt = kron(w_y_opt, w_z_opt);
for sc = 1:num_sc
    powGain_dB_benchmark_3(sc) = 10*log10(real(trace(CCM(:,:,sc) * (w_opt * w_opt'))));
end

powGain_dB_benchmark_1;
find(powGain_dB_benchmark_2 - powGain_dB_benchmark_1 > 1e-10)
find(powGain_dB_benchmark_3 - powGain_dB_benchmark_1 > 1e-10)
