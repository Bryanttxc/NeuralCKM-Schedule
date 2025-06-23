% created by Bryanttxc, FUNLab

function [cur_minSE, matches_result, tmp_rho_alloc] = max_minSE_inSlot(config, tmp_UE_group, tmp_rho_alloc, SE_tb_matrix_const, ...
                                                                        numUE_perSlot, num_IRS, num_served, num_UE, slot)
%BENCH_4_2 benchmark for algorithm AO optimization
% %%% input %%%
% tmp_UE_group: UE combination in current slot
% SE_tb_matrix_const: ergodic SE matrix, dim: num_UE x (num_IRS + 1)
% numUE_perSlot: number of UE within a slot
% num_IRS: number of IRSs
% num_served: number of served
% %%% output %%%
% cur_minSE: min ergodic SE in current slot
% matches_result: match result in current slot
% tmp_rho_alloc: rho allocation in current slot

SE_matrix_inSlot = SE_tb_matrix_const.Value(tmp_UE_group,:) .* tmp_rho_alloc;

if config.use_bench_4_2_enum == 1 % benchmark 1: enum + LP

    UE_id_inSlot = 1:numUE_perSlot;
    IRS_ids_inSlot = 1:num_IRS;
    combs = nchoosek(UE_id_inSlot, num_IRS); % UE combination served by IRS
    cur_minSE = -inf;
    
    for i = 1:size(combs,1)
        IRS_serve_UE = combs(i,:);
        BS_serve_UE = setdiff(UE_id_inSlot, IRS_serve_UE); % rest UEs served by BS
    
        perms_irs = perms(IRS_ids_inSlot); % 对IRS编号做全排列
        for j = 1:size(perms_irs, 1)
            assign = zeros(numUE_perSlot, 1);
            assign(BS_serve_UE) = num_served; % BS-served
            for k = 1:num_IRS
                assign(IRS_serve_UE(k)) = perms_irs(j,k); % IRS-served
            end
    
            % 获取每个用户在选定服务源下的速率
            sel_SE_tb = zeros(numUE_perSlot, 1);
            for u = 1:numUE_perSlot
                UE_real_id = tmp_UE_group(u);
                sel_SE_tb(u) = SE_tb_matrix_const.Value(UE_real_id, assign(u));
            end
    
            % 求解线性规划: max t, s.t. alpha_i * r_i >= t, sum(rho)=1, rho>=0
            rho = optimvar('rho',numUE_perSlot,'LowerBound',0);
            t = optimvar('t','LowerBound',0);
            prob = optimproblem('Objective',t,'ObjectiveSense','maximize');
            prob.Constraints.cons1 = sum(rho) == 1;
            for u = 1:numUE_perSlot
                prob.Constraints.(['c',num2str(u)]) = rho(u)*sel_SE_tb(u) >= t;
            end
    
            opts = optimoptions('linprog','Display','off');
            [sol,fval,exitflag] = solve(prob,'Options',opts);
    
            if exitflag == 1 && sol.t > cur_minSE
                cur_minSE = sol.t;
                matches_result = assign;
                tmp_rho_alloc = sol.rho;
            end
        end
    end

elseif config.use_bench_4_2_greedy == 1 % benchmark 2: greedy + LP
    
    % initial
    UE_matched = false(1, numUE_perSlot);
    IRS_matched = false(1, num_IRS);
    assign = ones(numUE_perSlot, 1) * num_served;
    cur_minSE = -inf;

    % create (UE_id, IRS_id, sel_SE) pair
    pairs = [];
    for u = 1:numUE_perSlot
        UE_real_id = tmp_UE_group(u);
        for r = 1:num_IRS
            pairs = [pairs; u, r, SE_tb_matrix_const.Value(UE_real_id, r)];
        end
    end

    % sort in descend order
    [~, idx] = sort(pairs(:,3), 'descend');
    pairs = pairs(idx,:);

    % greedy match
    for k = 1:size(pairs,1)
        u = pairs(k,1);
        r = pairs(k,2);
        if ~UE_matched(u) && ~IRS_matched(r)
            assign(u) = r;
            UE_matched(u) = true;
            IRS_matched(r) = true;
        end
    end

    % 获取每个用户在选定服务源下的速率
    sel_SE_tb = zeros(numUE_perSlot, 1);
    for u = 1:numUE_perSlot
        UE_real_id = tmp_UE_group(u);
        sel_SE_tb(u) = SE_tb_matrix_const.Value(UE_real_id, assign(u));
    end

    % 求解线性规划: max t, s.t. alpha_i * r_i >= t, sum(rho)=1, rho>=0
    rho = optimvar('rho',numUE_perSlot,'LowerBound',0);
    t = optimvar('t','LowerBound',0);
    prob = optimproblem('Objective',t,'ObjectiveSense','maximize');
    prob.Constraints.cons1 = sum(rho) == 1;
    for u = 1:numUE_perSlot
        prob.Constraints.(['c',num2str(u)]) = rho(u)*sel_SE_tb(u) >= t;
    end

    opts = optimoptions('linprog','Display','off');
    [sol,fval,exitflag] = solve(prob,'Options',opts);

    if exitflag == 1 && sol.t > cur_minSE
        cur_minSE = sol.t;
        matches_result = assign;
        tmp_rho_alloc = sol.rho;
    end

elseif config.use_bench_4_2_Hungarian == 1 % benchmark 3: Hungarian + LP

    cur_minSE = -inf;
    W = SE_tb_matrix_const.Value(tmp_UE_group, 1:num_IRS); % 不含基站
    costMatrix = -W; % matchpairs是最小权匹配，取负变最大
    
    % 2. 使用 matchpairs 求最大权匹配
    [matches, ~] = matchpairs(costMatrix, 1e6); % 每个 IRS、用户最多配一个
    
    % 3. 构建服务源assign，初始都为num_served（基站）
    assign = ones(numUE_perSlot, 1) * num_served;
    for k = 1:size(matches,1)
        UE = matches(k,1); % 行索引 → 用户编号
        IRS = matches(k,2); % 列索引 → IRS编号
        assign(UE) = IRS;
    end
    
    % 4. 获取每个用户在选定服务源下的速率
    sel_SE_tb = zeros(numUE_perSlot, 1);
    for u = 1:numUE_perSlot
        UE_real_id = tmp_UE_group(u);
        sel_SE_tb(u) = SE_tb_matrix_const.Value(UE_real_id, assign(u));
    end
    
    % 5. 求解线性规划: max t, s.t. alpha_i * r_i >= t, sum(rho)=1, rho>=0
    rho = optimvar('rho',numUE_perSlot,'LowerBound',0);
    t = optimvar('t','LowerBound',0);
    prob = optimproblem('Objective',t,'ObjectiveSense','maximize');
    prob.Constraints.cons1 = sum(rho) == 1;
    for u = 1:numUE_perSlot
        prob.Constraints.(['c',num2str(u)]) = rho(u)*sel_SE_tb(u) >= t;
    end
    
    opts = optimoptions('linprog','Display','off');
    [sol,fval,exitflag] = solve(prob,'Options',opts);
    
    if exitflag == 1 && sol.t > cur_minSE
        cur_minSE = sol.t;
        matches_result = assign;
        tmp_rho_alloc = sol.rho;
    end

else % Proposed scheme

    pre_minSE = 0;
    cur_minSE = 0;
    warn_cnt = 0;
    matches_result = zeros(1, numUE_perSlot);

    converge_gap = config.converge_gap;
    error_tolerate = config.phase1_tolerate;

    % -------------------- Proposed: AO optimization -------------------- %
    while 1

        % ----- Gale-Shapley algorithm (P2-1)----- %
        % step1: build up preference List
        % 1. IRS组preferences排序应按照"接受BS服务时的用户SE"升序排，故各irs的preferList应都相同
        % 2. IRS用于帮助BS服务不到的UE，所以把SE低的优先服务，这样拉高的用户最低SE
        % 3. UE组的preferences排序应按照"用户SE"降序排
        [~, ascend_BS_serve_UE_id] = sort(SE_matrix_inSlot(:,end), 1, 'ascend');
        IRS_preferlist = repmat(ascend_BS_serve_UE_id', num_IRS, 1); % dim: numIRS x numUE
        [~, UE_preferlist] = sort(SE_matrix_inSlot(:,1:end-1), 2, 'descend'); % dim: numUE x numIRS

        % step2: execute algorithm
        matches_result = one2oneMatching(IRS_preferlist, UE_preferlist); % dim: 1 x numUE_perSlot

        % ----- Iterative Balancing algorithm (P2-2)----- %
        % step1: make sure SE of each UE after finishing matching
        tmp_id = sub2ind(size(SE_matrix_inSlot), (1:numUE_perSlot), matches_result);
        SE_inSlot = SE_matrix_inSlot(tmp_id);

        % step2: execute algorithm
        [opt_rho, SE_inSlot] = IterativeBalancing(tmp_rho_alloc, SE_inSlot);

        % ----- check if meets convergence ----- %
        cur_minSE = min(SE_inSlot);
        meet_warn = cur_minSE - pre_minSE < 0 & abs(cur_minSE - pre_minSE) > 1e-15;
        meet_converge = cur_minSE - pre_minSE < converge_gap;
        if meet_warn
            warn_cnt = warn_cnt + 1;
            fprintf("[Warning] numIRS=%d, numUE=%d, Slot %d: " + ...
                "This iteration is detected to be less than the previous iteration\n", ...
                num_IRS, num_UE, slot);
            if warn_cnt >= error_tolerate
                break;
            end
        elseif meet_converge
            % fprintf("[Info] Slot %d: Meet Convergence! minimum Spectrum Efficiency = %.4f\n", slot, cur_minSE);
            SE_matrix_inSlot = SE_matrix_inSlot ./ tmp_rho_alloc .* opt_rho;
            tmp_rho_alloc = opt_rho;
            break;
        else
            % update variables
            SE_matrix_inSlot = SE_matrix_inSlot ./ tmp_rho_alloc .* opt_rho;
            tmp_rho_alloc = opt_rho;
            pre_minSE = cur_minSE;
            warn_cnt = 0;
        end

    end % end while

end

end
