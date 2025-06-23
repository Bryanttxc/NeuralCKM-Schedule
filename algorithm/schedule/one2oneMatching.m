% created by Bryanttxc, FUNLab

function [match_result] = one2oneMatching(A_preferList,B_preferList)
%ONE2ONEMATCHING Modified Gale-Shapley algorithm for Problem (P2-1)
%   create the stable match pair between A and B within a slot
%   computation complexity: num_A x num_B
%
% %%% input %%%
% A_preferList, dim: num_A x num_B, val: B_index
% B_preferList, dim: num_B x num_A, val: A_index
% %%% output %%%
% match_result, dim: 1 x num_B, val: A_index

num_A = size(A_preferList, 1);
num_B = size(B_preferList, 1);

if num_A > num_B
    error("[Info] 1st dimension of A_preferList must not less than that of B_preferList\n");
end

B_partner = zeros(1, num_B); % currect match result，0 means unmatched
A_proposals = zeros(num_A, num_B); % record proposal from A to B
free_A = 1:num_A; % all unmatched A

while ~isempty(free_A)

    % select one in A
    select_A_idx = free_A(1);
    free_A(1) = []; % remove idx

    % find the select_b_idx that select_A_idx wants to match the most
    for No_prefer = 1:num_B
        select_B_idx = A_preferList(select_A_idx, No_prefer);
        % 0 means no request before
        % 1 means requested and has been dumped later
        if A_proposals(select_A_idx, select_B_idx) == 0
            break;
        end
    end

    % select_A_idx proposes to select_b_idx
    A_proposals(select_A_idx, select_B_idx) = 1;

    % check select_b_idx response
    current_partner = B_partner(select_B_idx);
    if current_partner == 0
        % if not match, select_b_idx accept the proposal
        B_partner(select_B_idx) = select_A_idx;
    else
        % if has matched, determine which one prefers
        preferList = B_preferList(select_B_idx, :);
        if find(preferList == select_A_idx) < find(preferList == current_partner)
            % prefer to match select_A_idx, update
            B_partner(select_B_idx) = select_A_idx;
            free_A = [free_A, current_partner];
        else
            % prefer to match currect_partner, reject
            free_A = [free_A, select_A_idx];
        end
    end

end

match_result = B_partner;
match_result(match_result == 0) = num_A + 1; % served by BS if need

end
