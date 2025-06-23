% created by Bryanttxc, FUNLab

function [opt_UE_alloc, opt_rho_alloc, opt_UE_IRS_matches, minSE_inSlot] = ...
                            max_minSE_acrossSlot(config, init_UE_alloc_matrix, init_rho_alloc_matrix, UE_IRS_matches, ...
                                                 SE_tb_matrix_const, minSE_inSlot, numUE_perSlot, num_IRS, num_UE)
%UNTITLED benchmark for algorithm UE initial arrangement
%   此处显示详细说明

num_BS = config.num_BS;
num_served = num_BS + num_IRS;
num_slot = config.num_slot;

N_max = config.N_max; % max iterations
N = 1;

if config.use_bench_4_3_greedy == 1 % benchmark: greedy

    opt_UE_alloc = init_UE_alloc_matrix;
    opt_rho_alloc = init_rho_alloc_matrix;
    opt_UE_IRS_matches = UE_IRS_matches;
    best_min_SE = min(minSE_inSlot); % current global minSE
    
    while N <= N_max
    
        improved = false;
        tmp_UE_alloc_matrix = init_UE_alloc_matrix;
        
        % find worst_slot（corresponding to min_minSE）
        [~, worst_slot] = min(minSE_inSlot);
        
        % transfer UEs
        for cur_slot = 1:num_slot
    
            if cur_slot == worst_slot
                continue;
            end
            
            % exchange a pair of UEs between worst_slot and cur_slot
            for u1 = 1:numUE_perSlot
                for u2 = 1:numUE_perSlot

                    new_UE_alloc_matrix = tmp_UE_alloc_matrix;
                    
                    % transfer UE1 & UE2 in worst_slot & cur_slot
                    min_slot_UE_id = sub2ind(size(new_UE_alloc_matrix), worst_slot, u1);
                    cur_slot_UE_id = sub2ind(size(new_UE_alloc_matrix), cur_slot, u2);
                    new_UE_alloc_matrix([min_slot_UE_id cur_slot_UE_id]) = new_UE_alloc_matrix([cur_slot_UE_id min_slot_UE_id]);
                    
                    % update min_SE of cur_slot & worst_slot
                    slot_id = [worst_slot, cur_slot];
                    new_minSE_inSlot = minSE_inSlot;
                    new_UE_IRS_matches = UE_IRS_matches;
                    new_rho_alloc_matrix = init_rho_alloc_matrix;
                    for cur = 1:length(slot_id)
                        slot = slot_id(cur);
                        tmp_UE_group = new_UE_alloc_matrix(slot,:);
                        tmp_rho_alloc = init_rho_alloc_matrix(slot,:)';
                        [new_minSE_inSlot(slot,1), new_UE_IRS_matches(slot,:), new_rho_alloc_matrix(slot,:)] = ...
                                                            max_minSE_inSlot(config, tmp_UE_group, tmp_rho_alloc, SE_tb_matrix_const, ...
                                                                             numUE_perSlot, num_IRS, num_served, num_UE, slot);
                    end
                    
                    % update UE combinations if improve
                    new_min_SE = min(new_minSE_inSlot);
                    if new_min_SE > best_min_SE
                        best_min_SE = new_min_SE;
                        opt_UE_alloc = new_UE_alloc_matrix;
                        opt_rho_alloc = new_rho_alloc_matrix;
                        opt_UE_IRS_matches = new_UE_IRS_matches;
                        minSE_inSlot = new_minSE_inSlot;
                        improved = true;
                    end

                end
            end

            tmp_UE_alloc_matrix = opt_UE_alloc;
            init_rho_alloc_matrix = opt_rho_alloc;
            UE_IRS_matches = opt_UE_IRS_matches;
        end
        
        % quit if not improve
        if ~improved
            break;
        end
        
        init_UE_alloc_matrix = tmp_UE_alloc_matrix;
        N = N + 1;
    end

    opt_UE_alloc = init_UE_alloc_matrix;

elseif config.use_bench_4_3_enhanced == 1 % enhanced proposed version 

    converge_gap = config.converge_gap; % can adjust
    best_min_SE = min(minSE_inSlot); % current global minSE

    while N <= N_max
    
        tmp_UE_alloc_matrix = init_UE_alloc_matrix;
        improved = false;

        % find the largest SE gap corresponding two slots
        [min_minSE, minSlot_id] = min(minSE_inSlot);
        [max_minSE, maxSlot_id] = max(minSE_inSlot);

        % check if end iteration
        max_gap = max_minSE - min_minSE;
        if max_gap < converge_gap
            break;
        end

        % exchange a pair of UEs between worst_slot and cur_slot
        for u1 = 1:numUE_perSlot
            for u2 = 1:numUE_perSlot

                new_UE_alloc_matrix = tmp_UE_alloc_matrix;
                
                % transfer UE1 & UE2 in worst_slot & cur_slot
                min_slot_UE_id = sub2ind(size(new_UE_alloc_matrix), minSlot_id, u1);
                cur_slot_UE_id = sub2ind(size(new_UE_alloc_matrix), maxSlot_id, u2);
                new_UE_alloc_matrix([min_slot_UE_id cur_slot_UE_id]) = new_UE_alloc_matrix([cur_slot_UE_id min_slot_UE_id]);
                
                % update min_SE of cur_slot & worst_slot
                slot_id = [minSlot_id, maxSlot_id];
                new_minSE_inSlot = minSE_inSlot;
                new_UE_IRS_matches = UE_IRS_matches;
                new_rho_alloc_matrix = init_rho_alloc_matrix;
                for cur = 1:length(slot_id)
                    slot = slot_id(cur);
                    tmp_UE_group = new_UE_alloc_matrix(slot,:);
                    tmp_rho_alloc = init_rho_alloc_matrix(slot,:)';
                    [new_minSE_inSlot(slot,1), new_UE_IRS_matches(slot,:), new_rho_alloc_matrix(slot,:)] = ...
                                                        max_minSE_inSlot(config, tmp_UE_group, tmp_rho_alloc, SE_tb_matrix_const, ...
                                                                         numUE_perSlot, num_IRS, num_served, num_UE, slot);
                end
                
                % update UE combinations if improve
                new_min_SE = min(new_minSE_inSlot);
                if new_min_SE > best_min_SE
                    best_min_SE = new_min_SE;
                    init_UE_alloc_matrix = new_UE_alloc_matrix;
                    init_rho_alloc_matrix = new_rho_alloc_matrix;
                    UE_IRS_matches = new_UE_IRS_matches;
                    minSE_inSlot = new_minSE_inSlot;
                    improved = true;
                end

            end
        end

        % quit if not improve
        if ~improved
            break;
        end

        N = N + 1;

    end % end while

    opt_UE_alloc = init_UE_alloc_matrix;
    opt_rho_alloc = init_rho_alloc_matrix;
    opt_UE_IRS_matches = UE_IRS_matches;

elseif config.use_4_3_enhanced == 1 % revised proposed version 

    converge_gap = config.converge_gap; % can adjust
    best_min_SE = min(minSE_inSlot); % current global minSE

    while N <= N_max
    
        tmp_UE_alloc_matrix = init_UE_alloc_matrix;
        tmp_rho_alloc_matrix = init_rho_alloc_matrix;
        improved = false;

        % find the largest SE gap corresponding two slots
        [min_minSE, minSlot_id] = min(minSE_inSlot);
        [max_minSE, maxSlot_id] = max(minSE_inSlot);

        % check if end iteration
        max_gap = max_minSE - min_minSE;
        if max_gap < converge_gap
            break;
        end

        % select the worst UE in worst_slot
        rho_alloc_minSlot = tmp_rho_alloc_matrix(minSlot_id,:);
        [~, maxRho_UE_id_minSlot] = max(rho_alloc_minSlot);
        u1 = UE_IRS_matches(minSlot_id, maxRho_UE_id_minSlot);

        % exchange a pair of UEs between worst_slot and cur_slot
        for u2 = 1:numUE_perSlot

            new_UE_alloc_matrix = tmp_UE_alloc_matrix;
            
            % transfer UE1 & UE2 in worst_slot & cur_slot
            min_slot_UE_id = sub2ind(size(new_UE_alloc_matrix), minSlot_id, u1);
            cur_slot_UE_id = sub2ind(size(new_UE_alloc_matrix), maxSlot_id, u2);
            new_UE_alloc_matrix([min_slot_UE_id cur_slot_UE_id]) = new_UE_alloc_matrix([cur_slot_UE_id min_slot_UE_id]);
            
            % update min_SE of cur_slot & worst_slot
            slot_id = [minSlot_id, maxSlot_id];
            new_minSE_inSlot = minSE_inSlot;
            new_UE_IRS_matches = UE_IRS_matches;
            new_rho_alloc_matrix = init_rho_alloc_matrix;
            for cur = 1:length(slot_id)
                slot = slot_id(cur);
                tmp_UE_group = new_UE_alloc_matrix(slot,:);
                tmp_rho_alloc = init_rho_alloc_matrix(slot,:)';
                [new_minSE_inSlot(slot,1), new_UE_IRS_matches(slot,:), new_rho_alloc_matrix(slot,:)] = ...
                                                    max_minSE_inSlot(config, tmp_UE_group, tmp_rho_alloc, SE_tb_matrix_const, ...
                                                                     numUE_perSlot, num_IRS, num_served, num_UE, slot);
            end
            
            % update UE combinations if improve
            new_min_SE = min(new_minSE_inSlot);
            if new_min_SE > best_min_SE
                best_min_SE = new_min_SE;
                init_UE_alloc_matrix = new_UE_alloc_matrix;
                init_rho_alloc_matrix = new_rho_alloc_matrix;
                UE_IRS_matches = new_UE_IRS_matches;
                minSE_inSlot = new_minSE_inSlot;
                improved = true;
            end

        end

        % quit if not improve
        if ~improved
            break;
        end

        N = N + 1;

    end % end while

    opt_UE_alloc = init_UE_alloc_matrix;
    opt_rho_alloc = init_rho_alloc_matrix;
    opt_UE_IRS_matches = UE_IRS_matches;    

else % Proposed origin version (faster but lower accuracy)

    converge_gap = config.converge_gap; % can adjust
    tolerate_max = config.phase2_tolerate; % can adjust
    fail_cnt = 0;

    while N <= N_max
    
        tmp_UE_alloc_matrix = init_UE_alloc_matrix;
        tmp_rho_alloc_matrix = init_rho_alloc_matrix;

        % find the largest SE gap corresponding two slots
        [min_minSE, minSlot_id] = min(minSE_inSlot);
        [max_minSE, maxSlot_id] = max(minSE_inSlot);

        % check if end iteration
        max_gap = max_minSE - min_minSE;
        if max_gap < converge_gap
            break;
        end

        % tranfer two selected UEs in corresponding two slots
        % --------------------- scheme1 --------------------- %
        % bottleneck UE in minSlot with bigger rho -> lower SE under same rho
        % no change the matching relation between UEs and IRSs
        rho_alloc_minSlot = tmp_rho_alloc_matrix(minSlot_id,:);
        pick_swap_IRS_id = -1;
        fail = fail_cnt;
        while sum( ismember(rho_alloc_minSlot,-1) ) < numUE_perSlot
            [~, maxRho_UE_id_minSlot] = max(rho_alloc_minSlot);
            rho_alloc_minSlot(maxRho_UE_id_minSlot) = -1;
            tmp_id = UE_IRS_matches(minSlot_id, maxRho_UE_id_minSlot);
            if tmp_id ~= num_served % exclude BS-served
                if fail == 0
                    pick_swap_IRS_id = tmp_id;
                    % fprintf("[Info] minSlot %d choose UE %d-IRS %d\n", ...
                    %       minSlot_id, maxRho_UE_id_minSlot, select_IRS_id);
                    break;
                else
                    fail = fail - 1;
                end
            end
        end

        % check if no IRS-served UE can swap
        if pick_swap_IRS_id == -1 || fail_cnt > tolerate_max
            opt_UE_alloc = init_UE_alloc_matrix;
            opt_rho_alloc = init_rho_alloc_matrix;
            opt_UE_IRS_matches = UE_IRS_matches;
            break;
        end

        % --------------------- scheme2 --------------------- %
        % % random select
        % rand_order_id = randperm(numIRS);
        % select_IRS_id = rand_order_id(1);
        % fprintf("[Info] randomly choose IRS %d\n", select_IRS_id);

        % --------------------- Swap UEs --------------------- %
        pick_UE_pos_minSlot = find(UE_IRS_matches(minSlot_id, :) == pick_swap_IRS_id);
        pick_UE_pos_maxSlot = find(UE_IRS_matches(maxSlot_id, :) == pick_swap_IRS_id);

        min_slot_UE_id = sub2ind(size(tmp_UE_alloc_matrix), minSlot_id, pick_UE_pos_minSlot);
        max_slot_UE_id = sub2ind(size(tmp_UE_alloc_matrix), maxSlot_id, pick_UE_pos_maxSlot);
        tmp_UE_alloc_matrix([min_slot_UE_id max_slot_UE_id]) = tmp_UE_alloc_matrix([max_slot_UE_id min_slot_UE_id]);

        % update the rho allocation in corresponding two slots
        new_UE_rho_minSlot = tmp_rho_alloc_matrix(maxSlot_id, pick_UE_pos_maxSlot);
        new_UE_rho_maxSlot = tmp_rho_alloc_matrix(minSlot_id, pick_UE_pos_minSlot);

        old_UE_rho_minSlot = new_UE_rho_maxSlot;
        old_UE_rho_maxSlot = new_UE_rho_minSlot;

        tmp_rho_alloc_matrix(minSlot_id, [1:pick_UE_pos_minSlot-1, pick_UE_pos_minSlot+1:end]) = ...
             tmp_rho_alloc_matrix(minSlot_id, [1:pick_UE_pos_minSlot-1, pick_UE_pos_minSlot+1:end]) ./ (1-old_UE_rho_minSlot) .* (1-new_UE_rho_minSlot);

        tmp_rho_alloc_matrix(maxSlot_id, [1:pick_UE_pos_maxSlot-1, pick_UE_pos_maxSlot+1:end]) = ...
             tmp_rho_alloc_matrix(maxSlot_id, [1:pick_UE_pos_maxSlot-1, pick_UE_pos_maxSlot+1:end]) ./ (1-old_UE_rho_maxSlot) .* (1-new_UE_rho_maxSlot);

        tmp_rho_alloc_matrix(minSlot_id,pick_UE_pos_minSlot) = new_UE_rho_minSlot;
        tmp_rho_alloc_matrix(maxSlot_id,pick_UE_pos_maxSlot) = new_UE_rho_maxSlot;

        % enter Iterative Balancing
        minSE_afterSwap = zeros(2, 1);
        slot_id = [minSlot_id, maxSlot_id];
        for cur = 1:length(slot_id)
            slot = slot_id(cur);
            tmp_UE_group = tmp_UE_alloc_matrix(slot,:);
            tmp_rho_alloc = tmp_rho_alloc_matrix(slot,:)'; % PS: dimension not match revised
            SE_matrix_inSlot = SE_tb_matrix_const.Value(tmp_UE_group,:) .* tmp_rho_alloc;

            tmp_id = sub2ind(size(SE_matrix_inSlot), (1:numUE_perSlot), UE_IRS_matches(slot,:));
            SE_inSlot = SE_matrix_inSlot(tmp_id);

            [opt_rho, SE_inSlot] = IterativeBalancing(tmp_rho_alloc, SE_inSlot);
            tmp_rho_alloc_matrix(slot,:) = opt_rho;
            minSE_afterSwap(cur,:) = min(SE_inSlot);
        end

        % check if improve
        if min(minSE_afterSwap) > min_minSE ... % improve max-min
                && minSE_afterSwap(1) > min_minSE && minSE_afterSwap(2) < max_minSE ... % reduce gap
                % && minSE_afterSwap(2) >= minSE_afterSwap(1) % retain the order
            init_UE_alloc_matrix = tmp_UE_alloc_matrix;
            init_rho_alloc_matrix = tmp_rho_alloc_matrix;
            minSE_inSlot([minSlot_id, maxSlot_id], 1) = minSE_afterSwap;
            fail_cnt = 0;
        else
            fail_cnt = fail_cnt + 1;
        end

        N = N + 1;

    end % end while
    
end
