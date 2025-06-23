% created by Bryanttxc, FUNLab

function [Phi_opt, objval, minGain] = dc_sca_algorithm(ARV_y, ARV_z, Phi_y_init, Phi_z_init, RadiaPattern, num_sc, normalization, index)
% DC_SCA_ALGORITHM calculate the sub-optimal phase coefficients in R-axis,
% index = 'y' or 'z'

%% initialization
if index == 'y'
    ARV_iter = ARV_y;
    Phi_iter = Phi_y_init;
    ARV_other = ARV_z;
    Phi_other = Phi_z_init;
elseif index == 'z'
    ARV_iter = ARV_z;
    Phi_iter = Phi_z_init;
    ARV_other = ARV_y;
    Phi_other = Phi_y_init;
end
M_iter = size(Phi_iter,1);
M_other = size(Phi_other,1);

[U_W_k,S_W_k,~] = svd(Phi_iter);
spectral_norm = max(max(S_W_k));
max_singvec = U_W_k(:,1);

penalty = 20;
sumition_other = zeros(num_sc,1);

%% CVX
cvx_solver mosek
% cvx_save_prefs
cvx_begin
    variable Phi(M_iter,M_iter) complex semidefinite
    variable minGain
    expression constraints(num_sc,1)
    expression sumition_iter(num_sc,1)

    Phi_reshape = reshape(Phi,[M_iter*M_iter,1]);
    Phi_other_reshape = reshape(Phi_other,[M_other*M_other,1]);
    for sc = 1:num_sc
        covariance_iter = conj(ARV_iter(:,sc)) * ARV_iter(:,sc).';
        covariance_iter = reshape(covariance_iter.',[1,M_iter*M_iter]);
        sumition_iter(sc) = real(covariance_iter * Phi_reshape);

        covariance_other = conj(ARV_other(:,sc)) * ARV_other(:,sc).';
        covariance_other = reshape(covariance_other.',[1,M_other*M_other]);
        sumition_other(sc) = real(covariance_other * Phi_other_reshape);
    end
    constraints = sumition_iter .* sumition_other .* RadiaPattern / normalization;
    
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