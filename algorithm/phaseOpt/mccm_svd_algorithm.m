% created by Bryanttxc, FUNLab

function [phi_MCCM_SVD, instant_MCCM] = mccm_svd_algorithm(H_TxRIS_f, H_RISRx_f)
%MCCM_SVD_ALGORITHM AIRS phase coefficients optimized by MCCM-SVD scheme

%% Phase coefficient
% H_TxRIS_f: BS-AIRS link, dim: IRS_ant x BS_ant x num_sc
% H_RISRx_f: AIRS-UE link, dim: UE_ant x IRS_ant x num_sc

H_TxRIS_f = gpuArray(H_TxRIS_f); % gpu calculate
H_RISRx_f = gpuArray(H_RISRx_f);
instant_CCM = permute(H_RISRx_f, [2,1,3]) .* (H_TxRIS_f .* permute(conj(H_TxRIS_f), [2,1,3])) .* conj(H_RISRx_f);
instant_MCCM = mean(instant_CCM, 3);
instant_MCCM = double(gather(instant_MCCM));

[U_inst_mccm,~,~] = svd(instant_MCCM);
phi_MCCM_SVD = exp(-1j*(angle(U_inst_mccm(:,1)))); % phase extraction / normalization

end
