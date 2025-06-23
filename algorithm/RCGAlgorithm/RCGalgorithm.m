% created by Bryanttxc, FUNLab

function [phi_opt,meet_accuracy,two_norm_RG,objval_arr] = RCGalgorithm(CCM, accuracy)
% RCGALGORITHM optimize AIRS n phase coefficient to serve UE u implemented 
% by Riemannian conjugate gradient (RCG) algorithm 

%% Operation
M = size(CCM,1); % num of AIRS elements
funcTwoNorm = @(var)sqrt(sum(abs(var).^2)); % 2-norm operation
funcUnt = @(var)var./abs(var); % unt operation cited by [TaoMeiXia]
funcT = @(dir,phi)dir - real(dir.*conj(phi)) .* phi; % transport operation

% decrease complexity
index_irs = repmat(1:M,1,M);
index_irs(1:M+1:M^2) = [];
index_irs = reshape(index_irs',M-1,M);
index_irs = index_irs';

%% Initialize
phi = exp(1j*2*pi*rand(M,1));
RG_pre = calRiemannianGradient(CCM, phi, index_irs); % gradient
dir = -RG_pre; % direction
step = 1;
Armijo = 0.9; % adjust parameter
gamma = 0.5; % improve speed

%% RCG algorithm
count = 1;
meet_accuracy = true;
two_norm_RG = 0;
objval_arr = [];
while(funcTwoNorm(RG_pre) > accuracy)
    two_norm_RG = funcTwoNorm(RG_pre);
    
    % unable to solve
    if count > 500
        meet_accuracy = false;
        break;
    end
    
    % step 1 choose Armijo backtracking line search step size
    while( -calOBJ(CCM,funcUnt(phi+step*dir)) > -calOBJ(CCM,phi) + real(gamma*step*RG_pre'*dir) )
       step = step * Armijo;
       % avoid step too small
       if(step <= 1e-10)
           meet_accuracy = false;
           phi_opt = phi;
           return;
       end
    end
    
    objval_arr = [objval_arr calOBJ(CCM,funcUnt(phi+step*dir))];
    
    % step 2 update IRS phase coefficient
    phi = funcUnt(phi + step*dir);
    % step 3 update Riemannian gradient
    RG = calRiemannianGradient(CCM, phi, index_irs);
    % step 4 update search direction
    beta = RG'*(RG - RG_pre) / (RG_pre'*RG_pre); % Polak-Ribiere parameter
    dir = -RG + beta*funcT(dir,phi);
    % step 5 preserve RG
    RG_pre = RG;
    % step 6 record iterative count
    count = count + 1;
end

phi_opt = phi;
% plot(objval_arr,'linewidth',2)

end
