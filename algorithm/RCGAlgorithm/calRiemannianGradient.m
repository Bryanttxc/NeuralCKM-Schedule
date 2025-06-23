% created by Bryanttxc, FUNLab

function [RG] = calRiemannianGradient(CCM, phi, index_irs)
% CALRIEMANNIANGRADIENT calculate Riemannian Gradient

%% Euclidean Gradient
M = size(CCM,1);
num_sc = size(CCM,3);
EG_temp = zeros(M,num_sc);
for sc = 1:num_sc
    CCM_temp = CCM(:,:,sc);
    for m = 1:M
        temp = index_irs(m,:);
        EG_temp(m,sc) = -CCM_temp(m,temp) * phi(temp); % sum(CCM_temp(index_irs) .* phi(index_irs),2);
    end
end
EG = sum(EG_temp,2);

%% Riemannian Gradient
RG = EG - real(EG .* conj(phi)) .* phi;

end