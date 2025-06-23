% created by Bryanttxc, FUNLab

function [UE_alloc_matrix] = Preprocess(config, SE_matrix, numUE_perSlot, num_slot, cluster_label)
%PREPROCESS Obtain initial UE allocation in each slot based on
% K-means-constrained (cKmeans) algorithm and Gale-Shapley algorithm
% %%% input %%%
% SE_matrix: ergodic SE matrix, dim: num_UE x (num_IRS + 1)
% numUE_perSlot: number of UE within a slot
% num_slot: number of slot
% cluster_label: cluster result based on cKmeans
% %%% output %%%
% UE_alloc_matrix: initial UE allocation scheme in all slots

num_UE = size(SE_matrix, 1);
num_BS = 1; % assume num_BS = 1
num_IRS = size(SE_matrix, 2) - num_BS; 
BS_id = num_IRS + 1;
numUE_BS_served = numUE_perSlot - num_IRS;

if config.use_bench_4_1_random == 1 % benchmark 1: round robin + random

    rand_id = randperm(num_UE);
    UE_alloc_matrix = reshape(rand_id, num_slot, numUE_perSlot);

elseif config.use_bench_4_1_sort == 1 % benchmark 2: round robin + sort max_SE_perUE

    max_SE_vec = max(SE_matrix(:,1:end-1), [], 2);
    [~, UE_asc_order] = sort(max_SE_vec, 1, 'ascend');
    
    UE_alloc_matrix = reshape(UE_asc_order, num_slot, []);
    cols = 2:2:num_slot;
    UE_alloc_matrix(:, cols) = flipud(UE_alloc_matrix(:, cols));

else % proposed scheme

    %% rearrange cluster_label
    
    num_cluster = length(unique(cluster_label)); % assume equals to num_IRS
    cluster_label = reshape(cluster_label, 1, []); % make sure 1st dim = 1
    UE_id_vec = 1:num_UE;
    
    UE_label_comb = [UE_id_vec; cluster_label]';
    UE_label_comb = sortrows(UE_label_comb, 2, 'ascend'); % sorted by cluster-label
    
    UE_alloc_perCluster = cell(num_cluster, 1);
    for cluster_id = 1:num_cluster
        UE_alloc_perCluster{cluster_id} = UE_label_comb(UE_label_comb(:,2) == cluster_id-1, 1);
    end
    
    %% resolve conflict case
    
    % create priority matrix
    priority_matrix = zeros(num_IRS, num_cluster);
    for col = 1:num_IRS
        sel_UE_id = UE_alloc_perCluster{col};
        [~, idx] = max(SE_matrix(sel_UE_id, 1:num_IRS), [], 2);
        unique_idx = unique(idx);
        counts = histc(idx, unique_idx);
        priority_matrix(unique_idx, col) = counts;
    end
    
    % reset priority value if other columns are zero
    for row = 1:size(priority_matrix, 1)
        if sum(ismember( priority_matrix(row,:), 0 )) == num_IRS-1
            priority_matrix(row, priority_matrix(row,:)~=0) = num_IRS+1;
        end
    end
    
    % obtain cluster-IRS pair
    IRS_preferList = zeros(num_IRS, num_cluster);
    cluster_preferList = zeros(num_cluster, num_IRS);
    for cur = 1:num_IRS
        [~,IRS_preferList(cur,:)] = sort(priority_matrix(cur,:), 'descend');
        [~,cluster_preferList(cur,:)] = sort(priority_matrix(:,cur), 'descend');
    end
    IRS_serve_id_vec = one2oneMatching(IRS_preferList,cluster_preferList); % dim: 1 x num_cluster
    
    %% UE allocation in each slot
    
    UE_alloc_matrix = zeros(num_slot, num_IRS + numUE_BS_served);
    BS_served_UEs = []; % candidate BS-served UEs
    
    % determine IRS-served UEs
    for col = 1:length(IRS_serve_id_vec)
        IRS_id = IRS_serve_id_vec(col);
        sel_UE_id = UE_alloc_perCluster{col};
        % priority pick BS-served not good -> IRS-served
        [~, serve_order] = sortrows(SE_matrix(sel_UE_id,:), [BS_id IRS_id], {'ascend', 'descend'});
        % each IRS served one UE within a slot
        UE_alloc_matrix(:, IRS_id) = sel_UE_id(serve_order(1:num_slot));
        BS_served_UEs = [BS_served_UEs ; sel_UE_id(serve_order(num_slot+1:end))];
    end
    
    % determine BS-served UEs
    [~, serve_order] = sort(SE_matrix(BS_served_UEs, end), 'descend');
    UE_alloc_matrix(:, BS_id:end) = reshape(BS_served_UEs(serve_order), num_slot, []); % plan A
    % UE_alloc_inSlot(:, BS_id:end) = reshape(BS_serve_UEs(serve_order), numBS_serve, [])'; % plan B

end

end
