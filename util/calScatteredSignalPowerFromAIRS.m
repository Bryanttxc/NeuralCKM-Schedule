% created by Bryanttxc, FUNLab

function [coeff_nonserve] = calScatteredSignalPowerFromAIRS(ARV_i, ARV_r, F_n, G_n, E_G_n_u_l, varsigma_i_n, varsigma_r_n_u,...
                                                            RB, SCperRB, M, N, Lr, serve)

funcSingVal = @(ARV_r,Phi,ARV_i)sqrt( (ARV_r.'*Phi*ARV_i)' * (ARV_r.'*Phi*ARV_i) );
coeff_nonserve = zeros(RB,SCperRB,N); % coeff_nonserve = zeros(RB,SCperRB,N-1); not sure
for nonserve = [1:serve-1 serve+1:N]
    Phi_nonServe = diag(exp(1j*2*pi*rand(M,1)));
    for RB_idx = 1:RB
        for sc = 1: SCperRB
            singular_non = funcSingVal(ARV_r(:,:,sc,RB_idx,nonserve), Phi_nonServe, ARV_i(:,:,sc,RB_idx,nonserve));
            coeff_nonserve(RB_idx,sc,nonserve) = M^2/Lr*F_n(nonserve).^2.*G_n(nonserve).*E_G_n_u_l(nonserve).*varsigma_i_n(nonserve).*varsigma_r_n_u(nonserve).*singular_non^2;
        end
    end
end
coeff_nonserve = sum(coeff_nonserve,3);

end