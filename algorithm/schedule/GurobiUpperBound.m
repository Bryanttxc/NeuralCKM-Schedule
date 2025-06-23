% created by Bryanttxc, FUNLab

function [objvalue, runtime, opt_rho_Assign, opt_IRS_match] = GurobiUpperBound(numUE, numIRS, numSlot, SE_matrix, save_path, print_result)
% GUROBIUPPERBOUND use Gurobi to obtain the upper bound
% illustration：用户集合U，IRS集合N，时隙集合Q
% Optimization Model
%  maximize
%        R_min
%  subject to
%        R_u >= R_min, u \in U
%        0 <= rho_u_q <= 1, u \in U and q \in Q
%        sum_{u=1}^{U} rho_u_q = 1, q \in Q                    constraint1
%        sum_{u=1}^{U} mu_n_u_q = 1, n \in N, q \in Q          constraint2
%        sum_{n=1}^{N} mu_n_u_q <= 1, u \in U, q \in Q         constraint3
%        mu_n_u_q \in {0,1} binary
%        mu_d_u_q = 1-sum_{n=1}^{N} mu_n_u_q, u \in U, q \in Q constraint4
%        mu_d_u_q \in {0,1} binary
%        R_u = sum_{q=1}^{Q} [ mu_d_u_q*f_d_rho_u_q + ...
%                             \sum_{n=1}^{N}mu_n_u_q*f_n_rho_u_q ], u \in U
% optimized variables
% R_min, R_u, rho_u_q, mu_n_u_q, mu_d_u_q
% f_d_rho_u_q = f_d(rho_u_q)
% f_n_rho_u_q = f_n(rho_u_q, irs_phase)
% satisfy the assumption i and ii

addpath("D:\FunlabTool\Gurobi\win64\matlab")

%% Varname

Inf_val = 1e10;
endIdx = 1; % 指针,指向末尾
typeIndex = []; % 存储每个类别的末尾索引

% R_min -- 1
model.varnames{endIdx} = 'R_min';
model.lb(endIdx) = 0;
model.ub(endIdx) = +Inf;
model.vtype(endIdx) = 'C';
endIdx = endIdx + 1;
typeIndex = [typeIndex endIdx-1];

% R_u -- 2
for uu = 1:numUE
    model.varnames{endIdx} = sprintf('R_u_%d', uu);
    model.lb(endIdx) = 0;
    model.ub(endIdx) = +Inf;
    model.vtype(endIdx) = 'C';
    endIdx = endIdx + 1;
end
typeIndex = [typeIndex endIdx-1];

% rho_{u,q} dim=U*Q -- 3
for uu = 1:numUE
    for qq = 1:numSlot
        model.varnames{endIdx} = sprintf('rho_%d_%d', uu, qq);
        model.lb(endIdx) = 0;
        model.ub(endIdx) = 1;
        model.vtype(endIdx) = 'C';
        endIdx = endIdx + 1;
    end
end
typeIndex = [typeIndex endIdx-1];

% mu_{n,u,q} dim=N*U*Q -- 4
for nn = 1:numIRS
    for uu = 1:numUE
        for qq = 1:numSlot
            model.varnames{endIdx} = sprintf('mu_%d_%d_%d', nn, uu, qq);
            model.lb(endIdx) = 0;
            model.ub(endIdx) = 1;
            model.vtype(endIdx) = 'B';            
            endIdx = endIdx + 1;
        end
    end
end
typeIndex = [typeIndex endIdx-1];

% f_d(rho_{u,q}) dims=U*Q -- 5
for uu = 1:numUE
    for qq = 1:numSlot
        model.varnames{endIdx} = sprintf('f_d_rho_%d_%d', uu, qq);
        model.lb(endIdx) = 0;
        model.ub(endIdx) = +Inf;
        model.vtype(endIdx) = 'C';
        endIdx = endIdx + 1;
    end
end
typeIndex = [typeIndex endIdx-1];

% f_n(rho_{u,q}) dims=N*U*Q -- 6
for nn = 1:numIRS
    for uu = 1:numUE
        for qq = 1:numSlot
            model.varnames{endIdx} = sprintf('f_%d_rho_%d_%d', nn, uu, qq);
            model.lb(endIdx) = 0;
            model.ub(endIdx) = +Inf;
            model.vtype(endIdx) = 'C';
            endIdx = endIdx + 1;
        end
    end
end
typeIndex = [typeIndex endIdx-1];

% mu_f_n_rho_u_q = mu_{n,u,q}*f_n(rho_{u,q}) dims=N*U*Q -- 7
for nn = 1:numIRS
    for uu = 1:numUE
        for qq = 1:numSlot
            model.varnames{endIdx} = sprintf('mu_f_%d_rho_%d_%d', nn, uu, qq);
            model.lb(endIdx) = 0;
            model.ub(endIdx) = +Inf;
            model.vtype(endIdx) = 'C';
            endIdx = endIdx + 1;
        end
    end
end
typeIndex = [typeIndex endIdx-1];

% mu_{d,u,q} = 1-\sum_{n=1}^{N}mu_n_u_q dims=U*Q -- 8
for uu = 1:numUE
    for qq = 1:numSlot
        model.varnames{endIdx} = sprintf('mu_d_%d_%d', uu, qq);
        model.lb(endIdx) = 0;
        model.ub(endIdx) = 1;
        model.vtype(endIdx) = 'B';
        endIdx = endIdx + 1;
    end
end
typeIndex = [typeIndex endIdx-1];

% mu_f_d_rho_u_q = mu_d_u_q*f_d_rho_u_q dims=U*Q -- 9
for uu = 1:numUE
    for qq = 1:numSlot
        model.varnames{endIdx} = sprintf('mu_f_d_rho_%d_%d', uu, qq);
        model.lb(endIdx) = 0;
        model.ub(endIdx) = +Inf;
        model.vtype(endIdx) = 'C';
        endIdx = endIdx + 1;
    end
end
typeIndex = [typeIndex endIdx-1];

% num of variable
varLen = length(model.varnames);

%% Objective

model.modelsense = 'max';
model.obj = [1 zeros(1,varLen-1)];

%% Constraint

% typeIndex
%   1    2     3          4            5              6              7
% R_min R_u rho_{u,q} mu_{n,u,q} f_d(rho_{u,q}) f_n(rho_{u,q}) mu_f_n_rho_u_q
%     8             9
% mu_{d,u,q}  mu_f_d_rho_u_q

% constraint           1            2                3               4                  5                      6
tempMatrixA = zeros(numSlot + numIRS*numSlot + numUE*numSlot + numUE*numSlot + numIRS*numUE*numSlot + numIRS*numUE*numSlot + ...
                    numIRS*numUE*numSlot + numUE*numSlot + numUE*numSlot + numUE*numSlot + numUE, varLen);
%                            7                   8              9               10          11
sense = []; % symbol of inequality constraints
endIdx = 1; % reset

% constraint1: sum_{u=1}^{U} rho_u_q = 1, q \in Q
templen = typeIndex(3)-typeIndex(2); % U*Q
tempMatrix = zeros(numSlot,templen);
for nn = 1:numSlot
    tempMatrix(nn,nn:numSlot:templen) = 1;
end
tempMatrixA(endIdx:endIdx+numSlot-1,typeIndex(2)+1:typeIndex(3)) = tempMatrix;
sense = [sense; repmat('=', numSlot, 1)];
endIdx = endIdx + numSlot;

% constraint2: sum_{u=1}^{U} mu_n_u_q = 1, n \in N, q \in Q
templen = typeIndex(4)-typeIndex(3); % N*U*Q
tempMatrix = zeros(numIRS*numSlot,templen);
for nn = 1:numIRS
    for qq = 1:numSlot
        tempMatrix(qq+(nn-1)*numSlot,qq+(nn-1)*numUE*numSlot:numSlot:nn*numUE*numSlot) = 1;
    end
end
tempMatrixA(endIdx:endIdx+numIRS*numSlot-1,typeIndex(3)+1:typeIndex(4)) = tempMatrix;
endIdx = endIdx + numIRS*numSlot;
sense = [sense; repmat('=', numIRS*numSlot, 1)];

% constraint3: sum_{n=1}^{N} mu_n_u_q <= 1, u \in U, q \in Q
templen = typeIndex(4)-typeIndex(3); % N*U*Q
tempMatrix = zeros(numUE*numSlot,templen);
for uu = 1:numUE
    for qq = 1:numSlot
        tempMatrix(qq+(uu-1)*numSlot,qq+(uu-1)*numSlot:numUE*numSlot:templen) = 1;
    end
end
tempMatrixA(endIdx:endIdx+numUE*numSlot-1,typeIndex(3)+1:typeIndex(4)) = tempMatrix;
endIdx = endIdx + numUE*numSlot;
sense = [sense; repmat('<', numUE*numSlot, 1)];

% constraint4: mu_d_u_q <= 1 - \sum_{n=1}^{N}mu_n_u_q --> mu_d_u_q + \sum_{n=1}^{N}mu_n_u_q <= 1
templen = typeIndex(8)-typeIndex(3);
tempMatrix = zeros(numUE*numSlot,templen);
end_mu_n = typeIndex(4)-typeIndex(3);
start_mu_d = typeIndex(7)-typeIndex(3);
for uu = 1:numUE
    for qq = 1:numSlot
        tempMatrix(qq+(uu-1)*numSlot,qq+(uu-1)*numSlot:numUE*numSlot:end_mu_n) = 1; % \sum_{n=1}^{N}mu_n_u_q
        tempMatrix(qq+(uu-1)*numSlot,qq+(uu-1)*numSlot+start_mu_d) = 1; % mu_d_u_q
    end
end
tempMatrixA(endIdx:endIdx+numUE*numSlot-1,typeIndex(3)+1:typeIndex(8)) = tempMatrix;
endIdx = endIdx + numUE*numSlot;
sense = [sense; repmat('<', numUE*numSlot, 1)];

% constraint5: mu_f_n_rho_u_q <= +inf*mu_n_u_q --> -inf*mu_n_u_q + mu_f_n_rho_u_q <= 0
templen = typeIndex(4)-typeIndex(3);
for tt = 1:templen
    tempMatrixA(endIdx,typeIndex(3)+tt) = -Inf_val; % mu_n_u_q
    tempMatrixA(endIdx,typeIndex(6)+tt) = 1; % mu_f_n_rho_u_q
    endIdx = endIdx + 1;
end
sense = [sense; repmat('<', templen, 1)];

% constraint6: mu_f_n_rho_u_q <= f_n_rho_u_q --> -f_n_rho_u_q + mu_f_n_rho_u_q <= 0
templen = typeIndex(4)-typeIndex(3);
for tt = 1:templen
    tempMatrixA(endIdx,typeIndex(5)+tt) = -1; % f_n_rho_u_q
    tempMatrixA(endIdx,typeIndex(6)+tt) = 1; % mu_f_n_rho_u_q
    endIdx = endIdx + 1;
end
sense = [sense; repmat('<', templen, 1)];

% constraint7: mu_f_n_rho_u_q >= f_n_rho_u_q - inf*(1-mu_n_u_q) --> -inf*mu_n_u_q - f_n_rho_u_q + mu_f_n_rho_u_q >= -inf
% attention: rhs = -inf !!!
templen = typeIndex(4)-typeIndex(3);
for tt = 1:templen
    tempMatrixA(endIdx,typeIndex(3)+tt) = -Inf_val; % mu_n_u_q
    tempMatrixA(endIdx,typeIndex(5)+tt) = -1; % f_n_rho_u_q
    tempMatrixA(endIdx,typeIndex(6)+tt) = 1; % mu_f_n_rho_u_q
    endIdx = endIdx + 1;
end
sense = [sense; repmat('>', templen, 1)];

% constraint8: mu_f_d_rho_u_q <= +inf*mu_d_u_q --> -inf*mu_d_u_q + mu_f_d_rho_u_q <= 0
templen = typeIndex(9)-typeIndex(8);
for tt = 1:templen
    tempMatrixA(endIdx,typeIndex(7)+tt) = -Inf_val; % mu_d_u_q
    tempMatrixA(endIdx,typeIndex(8)+tt) = 1; % mu_f_d_rho_u_q
    endIdx = endIdx + 1;
end
sense = [sense; repmat('<', templen, 1)];

% constraint9: mu_f_d_rho_u_q <= f_d_rho_u_q --> -f_d_rho_u_q + mu_f_d_rho_u_q <= 0
templen = typeIndex(9)-typeIndex(8);
for tt = 1:templen
    tempMatrixA(endIdx,typeIndex(4)+tt) = -1; % f_d_rho_u_q
    tempMatrixA(endIdx,typeIndex(8)+tt) = 1; % mu_f_d_rho_u_q
    endIdx = endIdx + 1;
end
sense = [sense; repmat('<', templen, 1)];

% constraint10: mu_f_d_rho_u_q >= f_d_rho_u_q - inf*(1-mu_d_u_q) --> -inf*mu_d_u_q - f_d_rho_u_q + mu_f_d_rho_u_q >= -inf
% attention: rhs = -inf !!!
templen = typeIndex(9)-typeIndex(8);
for tt = 1:templen
    tempMatrixA(endIdx,typeIndex(7)+tt) = -Inf_val; % mu_d_u_q
    tempMatrixA(endIdx,typeIndex(4)+tt) = -1; % f_d_rho_u_q
    tempMatrixA(endIdx,typeIndex(8)+tt) = 1; % mu_f_d_rho_u_q
    endIdx = endIdx + 1;
end
sense = [sense; repmat('>', templen, 1)];

% constraint11: R_u = \sum_{q=1}^{Q} [ mu_f_d_rho_u_q + \sum_{n=1}^{N} mu_f_n_rho_u_q ]
% --> \sum_{q=1}^{Q}mu_f_d_rho_u_q + \sum_{q=1}^{Q}\sum_{n=1}^{N} mu_f_n_rho_u_q - R_u = 0
cur_mu_f_d = typeIndex(8)+1;
cur_mu_f_n = typeIndex(6)+1;
for uu = 1:numUE
    tempMatrixA(endIdx,typeIndex(1)+uu) = -1; % R_u
    tempMatrixA(endIdx,cur_mu_f_d:cur_mu_f_d+numSlot-1) = 1; % \sum_{q=1}^{Q} mu_f_d_rho_u_q
    
    idx = cur_mu_f_n;
    sel = [];
    for nn = 1:numIRS
        sel = [sel idx:idx+numSlot-1]; % \sum_{q=1}^{Q}
        idx = idx + numUE*numSlot; % \sum_{n=1}^{N}
    end
    tempMatrixA(endIdx,sel) = 1; % \sum_{q=1}^{Q}\sum_{n=1}^{N}mu_f_n_rho_u_q
    
    cur_mu_f_d = cur_mu_f_d + numSlot;
    cur_mu_f_n = cur_mu_f_n + numSlot;
    endIdx = endIdx + 1;
end
sense = [sense; repmat('=', numUE, 1)];

% summary
model.A = sparse(tempMatrixA);
model.sense = sense;
% constraint        1                     2                       3                       4                         5                                  6
model.rhs = [ones(numSlot,1) ; ones(numIRS*numSlot,1) ; ones(numUE*numSlot,1) ; ones(numUE*numSlot,1) ; zeros(numIRS*numUE*numSlot,1) ; zeros(numIRS*numUE*numSlot,1) ; ...
  -Inf_val*ones(numIRS*numUE*numSlot,1) ; zeros(numUE*numSlot,1) ; zeros(numUE*numSlot,1) ; -Inf_val*ones(numUE*numSlot,1) ; zeros(numUE,1)] ;
%              7                                    8                         9                         10                          11

%% General function constraints

% R_u >= R_min, u \in U
model.genconmin.vars = typeIndex(1)+1:typeIndex(2); % R_u
model.genconmin.resvar = 1; % R_min
model.genconmin.name = 'gcf0';

% f_d_rho_u_q
endIdx = 1;
for uu = 1:numUE
    for qq = 1:numSlot
        model.genconpoly(endIdx).xvar = typeIndex(2)+qq+(uu-1)*numSlot;
        model.genconpoly(endIdx).yvar = typeIndex(4)+qq+(uu-1)*numSlot;
        model.genconpoly(endIdx).p = [SE_matrix(uu,end), 0]; % kx
        model.genconpoly(endIdx).name = sprintf('gcf%d',endIdx);
        endIdx = endIdx + 1;
    end
end

% f_n_rho_u_q
for nn = 1:numIRS
    for uu = 1:numUE
        for qq = 1:numSlot
            model.genconpoly(endIdx).xvar = typeIndex(2)+qq+(uu-1)*numSlot;
            model.genconpoly(endIdx).yvar = typeIndex(5)+qq+(uu-1)*numSlot+(nn-1)*numUE*numSlot; % careful
            model.genconpoly(endIdx).p = [SE_matrix(uu,nn), 0]; % kx
            model.genconpoly(endIdx).name = sprintf('gcf%d',endIdx);
            endIdx = endIdx + 1;
        end
    end
end

%% Optimize

% set nonlinear property (can delete)
% params.FuncNonlinear = 1; % First approach: Set FuncNonlinear parameter
% model.genconmin.funcnonlinear = 1;
% for ttt = 1:length(model.genconpoly)
%     model.genconpoly(ttt).funcnonlinear = 1;
% end

gurobi_write(model, [save_path, '\optimal_MILP_cxt.lp']); % write result into file

params.outputflag = 0; % turn down output
params.mipgap = 1e-4; % gap
result = gurobi(model, params); % optimize

%% Result

runtime = result.runtime;

if strcmp(result.status, 'OPTIMAL')

    objvalue = result.objval;

    % SLOT  slot1    slot2
    % UE1  rho_1_1  rho_1_2
    % UE2  rho_2_1  rho_2_2
    % ...
    opt_rho_Assign = reshape( result.x(typeIndex(2)+1:typeIndex(3)), numSlot, numUE )';

    %        IRS1       IRS2       BS
    % UE1  mu_1_1_q   mu_2_1_q  mu_d_1_q
    % UE2  mu_1_2_q   mu_2_2_q  mu_d_2_q
    % ...(q = 1,..., numSlot, in 3-d)
    tempMatrx = reshape( result.x([typeIndex(3)+1:typeIndex(4), typeIndex(7)+1:typeIndex(8)]), numSlot, [] );
    for slot = 1:numSlot
        opt_IRS_match(:,:,slot) = reshape( tempMatrx(slot,:), numUE, []);
    end

    if print_result == 1
        fprintf('[Info] Caculate time of Gurobi: %.4f seconds\n', runtime)
        fprintf('[Info] Result is OPTIMAL!\n');
        fprintf('[Info] Optimal max-min Value: %.4f\n', objvalue);

        % Print in a easier recognize way
        slot_id = 1:numSlot;
        UE_id = 1:numUE;
        slot_name = compose('slot_%d', slot_id);
        UE_name = compose('UE_%d', UE_id);
        opt_rho_alloc_table = array2table(opt_rho_Assign, 'VariableNames', slot_name, 'RowNames', UE_name);
        content = evalc('disp(opt_rho_alloc_table)'); % 使用 evalc捕获disp的输出
        fprintf('[Info] Optimal rho allocation: \n %s', content);

        % Print in a easier recognize way
        IRS_id = 1:numIRS;
        UE_id = 1:numUE;
        IRS_name = compose('IRS_%d', IRS_id);
        UE_name = compose('UE_%d', UE_id);

        for slot = 1:numSlot
            fprintf('\n[Info] Optimal UE-IRS pair in slot %d: \n', slot);
            opt_IRS_match_inSlot = opt_IRS_match(:,:,slot);
            for serve_id = 1:numIRS+1
                UE_indices = find(opt_IRS_match_inSlot(:,serve_id) == 1 & opt_rho_Assign(:,slot) ~= 0);
                UE_name = "UE" + UE_indices;
                if serve_id == numIRS + 1
                    fprintf('[Info] BS-served: %s\n', char(join(UE_name)));
                else
                    fprintf('[Info] IRS%d-served: %s\n', serve_id, UE_name);
                end
            end

            % opt_IRS_match_table = array2table(opt_IRS_match(:,:,slot), 'VariableNames', [IRS_name, 'BS'], 'RowNames', UE_name);
            % content = evalc('disp(opt_IRS_match_table)'); % 使用 evalc捕获disp的输出
            % fprintf('[Info] Optimal UE-IRS pair in slot %d: \n %s', slot, content);
        end
    end

else
    fprintf('[Info] Failed! %s!\n', result.status);
end

end
