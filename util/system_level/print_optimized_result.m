% created by Bryanttxc, FUNLab

function print_optimized_result(scheme_name, scheme_level, time, ...
                                minSE, init_UE_alloc_matrix, opt_rho_alloc, opt_UE_IRS_matches)

num_BS = 1;
num_IRS = length(unique(opt_UE_IRS_matches(:))) - num_BS;
num_UE = length(opt_UE_IRS_matches(:));
num_slot = size(init_UE_alloc_matrix, 1);

% Print in a easier recognize way
fprintf("------------------------------------------------------------------------------\n");
fprintf("[Info] Caculate time of %s: %.4f seconds\n", scheme_name, time);
fprintf("[Info] Result is %s!\n", scheme_level);
fprintf("[Info] %s max-min Value: %.4f\n", scheme_level, minSE);

% --------------------- print rho assignment in each slot --------------------- %
opt_rho_Assign = zeros(num_UE, num_slot);
for slot = 1:num_slot
    opt_rho_Assign(init_UE_alloc_matrix(slot,:), slot) = opt_rho_alloc(slot,:)';
end
slot_id = 1:num_slot;
UE_id = 1:num_UE;
slot_name = compose('slot_%d', slot_id);
UE_name = compose('UE_%d', UE_id);
opt_rho_alloc_table = array2table(opt_rho_Assign, 'VariableNames', slot_name, 'RowNames', UE_name);
content = evalc('disp(opt_rho_alloc_table)'); % 使用 evalc捕获disp的输出
fprintf('[Info] %s rho allocation: \n %s', scheme_level, content);

% --------------------- print UE-IRS/BS match in each slot --------------------- %
opt_IRS_match = zeros(num_UE, num_IRS+num_BS, num_slot);
for slot = 1:num_slot
    tmp_match_mat = zeros(num_UE, num_IRS+1);
    idx = sub2ind([num_UE, num_IRS+1], init_UE_alloc_matrix(slot,:), opt_UE_IRS_matches(slot,:));
    tmp_match_mat(idx) = 1;
    opt_IRS_match(:,:,slot) = tmp_match_mat;
end
IRS_id = 1:num_IRS;
% UE_id = 1:numUE;
IRS_name = compose('IRS_%d', IRS_id);
UE_name = compose('UE_%d', UE_id);
for slot = 1:num_slot
    fprintf('\n[Info] %s UE-IRS pair in slot %d: \n', scheme_level, slot);
    opt_IRS_match_inSlot = opt_IRS_match(:,:,slot);
    for serve_id = 1:num_IRS+num_BS
        UE_indices = find(opt_IRS_match_inSlot(:,serve_id) == 1 & opt_rho_Assign(:,slot) ~= 0);
        UE_name = "UE" + UE_indices;
        if serve_id == num_IRS+num_BS
            fprintf('[Info] BS-served: %s\n', char(join(UE_name)));
        else
            fprintf('[Info] IRS%d-served: %s\n', serve_id, UE_name);
        end
    end

    % opt_IRS_match_table = array2table(opt_IRS_match(:,:,slot), 'VariableNames', [IRS_name, 'BS'], 'RowNames', UE_name);
    % content = evalc('disp(opt_IRS_match_table)'); % 使用 evalc捕获disp的输出
    % fprintf('[Info] %s UE-IRS pair in slot %d: \n %s', scheme_level, slot, content);
end

end
