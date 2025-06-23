% created by Bryanttxc, FUNLab

function [phi_opt, Phi_y_opt, Phi_z_opt] = AO_dc_sca_algorithm(ARV_y, ARV_z, phi_y_init, phi_z_init, RadiaPattern, num_sc)
% DC_SCA_ALGORITHM optimize phase coefficient by DC_SCA algorithm (CVX solver)
% cited by [1] https://arxiv.org/html/2505.01076v1

%% initialization

AO_iter = 10;
accuracy = 1e-3;
Phi_y_init = conj(phi_y_init) * phi_y_init.';
Phi_z_init = conj(phi_z_init) * phi_z_init.';
objval_pre = -200;

DC_y_arr = [];
reduce_y_arr = [];
DC_z_arr = [];
reduce_z_arr = [];

normalization_dB = 60; % Unit is 20dB
normalization = 10^(normalization_dB/10); % Unit is 1

%% Alternating optimization

tic
for iter = 1:AO_iter
    fprintf("第%d次AO迭代，优化y...\n",iter);
    while 1
        % SCA, fixed z
        [Phi_y_opt,objval_y,rho_y] = dc_sca_algorithm(ARV_y,ARV_z,Phi_y_init,Phi_z_init,RadiaPattern,num_sc,normalization,'y');

        % log
        DC_y_arr = [DC_y_arr 10*log10(rho_y*normalization) - objval_y];
        reduce_y_arr = [reduce_y_arr objval_y - objval_pre];
        fprintf("objval: %f\n", objval_y);
        fprintf('DC：%f\n',DC_y_arr(end));

        Phi_y_init = Phi_y_opt;
        objval_pre = objval_y;
        if reduce_y_arr(end) < accuracy
            break;
        end
    end
    
    fprintf("第%d次AO迭代，优化z...\n",iter);
    while 1
        % SCA, fixed y
        [Phi_z_opt,objval_z,rho_z] = dc_sca_algorithm(ARV_y,ARV_z,Phi_y_init,Phi_z_init,RadiaPattern,num_sc,normalization,'z');

        % log
        DC_z_arr = [DC_z_arr 10*log10(rho_z*normalization) - objval_z];
        reduce_z_arr = [reduce_z_arr objval_z - objval_pre];
        fprintf("objval: %f\n", objval_z);
        fprintf('DC：%f\n',DC_z_arr(end));

        Phi_z_init = Phi_z_opt;
        objval_pre = objval_z;
        if reduce_z_arr(end) < accuracy
            break;
        end
    end
end
toc

%% result

Phi_init = kron(Phi_y_init,Phi_z_init);
[U_init,S_init,~] = svd(Phi_init);
phi_opt = exp(1j*angle(U_init(:,1)));

Phi_y_opt = Phi_y_init;
Phi_z_opt = Phi_z_init;

end