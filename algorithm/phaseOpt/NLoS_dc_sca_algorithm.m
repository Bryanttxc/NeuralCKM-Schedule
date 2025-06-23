% created by Bryanttxc, FUNLab

function [Phi_opt, objval, minGain] = NLoS_dc_sca_algorithm(CCM, Phi_init, RadiaPattern, num_sc, normalization)
% DC_SCA_ALGORITHM calculate the sub-optimal phase coefficients in R-axis,
% index = 'y' or 'z'

%% initialization
[U_W_k,S_W_k,~] = svd(Phi_init);
spectral_norm = max(max(S_W_k));
max_singvec = U_W_k(:,1);

M_iter = length(Phi_init);
penalty = 20;

%% CVX
cvx_solver mosek
% cvx_save_prefs
cvx_begin
    variable Phi(M_iter,M_iter) complex semidefinite
    variable minGain
    expression constraints(num_sc,1)
    expression sumition_iter(num_sc,1)

    Phi_reshape = reshape(Phi,[M_iter*M_iter,1]);
    for sc = 1:num_sc
        covariance_iter = reshape(CCM(:,:,sc).',[1,M_iter*M_iter]);
        sumition_iter(sc) = real(covariance_iter * Phi_reshape);
    end
    constraints = sumition_iter .* RadiaPattern / normalization;
    
    nuclear_norm = norm_nuc(Phi);

    objval = 10*(-rel_entr(1,minGain*normalization))/log(10) ...
        - penalty * (nuclear_norm - ( spectral_norm + real(trace( (max_singvec*max_singvec') * (Phi-Phi_iter) )) )) / M_iter;
    
    maximize (objval)
    subject to
        constraints >= minGain;
        diag(Phi) == 1;
        Phi == hermitian_semidefinite(M_iter);
cvx_end

Phi_opt = Phi;

end