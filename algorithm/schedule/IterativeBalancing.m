% created by Bryanttxc, FUNLab

function [rho_opt, SE_u_opt] = IterativeBalancing(rho_vec, SE_u)
%ITERATIVEBALANCING Iterative balancing algorithm for Problem (P2-2)
%   allocate the spectrum resource of UEs within a slot
%   computation complexity: N_max * log2(1/converge_gap)
%
% %%% input %%%
% vec_rho: the proportion of rho for all UEs within a slot
% SE_u: ergodic spectrum efficiency of all UEs within a slot
% %%% output %%%
% rho_opt: the optimized proportion of rho for all UEs

converge_gap = 1e-3;
N_max = 500; % max number of iteration
N = 1; % counter

while N <= N_max

    % find the largest ergodic SE gap corresponding index
    [min_SE, min_idx] = min(SE_u);
    [max_SE, max_idx] = max(SE_u);

    % end iteration if satisfy
    if max_SE - min_SE < converge_gap
        break;
    end

    % tune the rho of max barSE and min barSE while keeping others fixed
    % max barSE -> turn down rho while min barSE -> turn up rho
    % SE_tb: SE with total bandwidth
    SE_tb_min = min_SE / rho_vec(min_idx); % 分子 1
    SE_tb_max = max_SE / rho_vec(max_idx); % 分母 3
    ideal_assign_scale = SE_tb_min / SE_tb_max; % 理想分配比例 1/3

    tmp_total_scale = rho_vec(max_idx) + rho_vec(min_idx); % 0.3 + 0.4 = 0.7
    ideal_rho_maxSE = tmp_total_scale * (SE_tb_min / (SE_tb_min + SE_tb_max)); % 0.7 * 1/4 = 7/40
    ideal_rho_minSE = tmp_total_scale * (SE_tb_max / (SE_tb_min + SE_tb_max)); % 0.7 * 3/4 = 21/40

    % plan_A and plan_B
    plan_A = ideal_rho_maxSE / ( rho_vec(min_idx) + (rho_vec(max_idx) - ideal_rho_maxSE) ); % (7/40) / (0.4 + (0.3-7/40))
    plan_B = ( rho_vec(max_idx) - (ideal_rho_minSE - rho_vec(min_idx)) ) / ideal_rho_minSE; % (0.3 - (21/40-0.4)) / (21/40)

    if abs(ideal_assign_scale - plan_A) < abs(ideal_assign_scale - plan_B)
        % choose plan_A
        rho_vec(min_idx) = rho_vec(min_idx) + (rho_vec(max_idx) - ideal_rho_maxSE);
        rho_vec(max_idx) = ideal_rho_maxSE;
    else
        % choose plan_B
        rho_vec(max_idx) = rho_vec(max_idx) - (ideal_rho_minSE - rho_vec(min_idx));
        rho_vec(min_idx) = ideal_rho_minSE;
    end

    % update variable
    SE_u(max_idx) = rho_vec(max_idx) * SE_tb_max;
    SE_u(min_idx) = rho_vec(min_idx) * SE_tb_min;
    N = N + 1;

end

rho_opt = rho_vec;
SE_u_opt = SE_u;

end
