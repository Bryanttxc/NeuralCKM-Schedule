% created by Bryanttxc, FUNLab

function schedule(config)
%SCHEDULE schedule algorithm design

% include paths of necessary folders
root_path = config.root_path;
addpath(fullfile(root_path, 'algorithm'));
addpath(fullfile(root_path, 'util'));
addpath(fullfile(root_path, 'util', 'system_level'));

% turn on parallel pool
turnOnParfor(config);

%% Preparation

save_data_path = config.save_data_path;
schedule_src_path = config.schedule_src_path;
py_interpreter = config.py_interpreter;

num_BS = config.num_BS;
num_IRS_vec = config.num_IRS_vec;
num_UE_vec = config.num_UE_vec;
num_slot = config.num_slot;
num_lab = config.num_lab;
max_num_sublab = config.max_num_sublab;

print_opt_result = config.print_opt_result;
default_create_file = config.create_new_input_file;
default_plot_graph = config.is_plot_graph;

%% Main Program

start_time = cputime;
for num_IRS_idx = 1:length(num_IRS_vec)

    num_IRS = num_IRS_vec(num_IRS_idx);
    num_served = num_IRS + num_BS; % number of servers
    config.create_new_input_file = default_create_file; % reset
    config.is_plot_graph = default_plot_graph; % reset

    %% Generate Sample

    input_file = fullfile(schedule_src_path, ['sys_level_data_numIRS_', num2str(num_IRS), '.mat']);
    py_UESENet_main_path = config.py_UESENet_main_path;

    % NN-1 sample
    nn_1_time = cputime;
    gen_sys_sample(config, input_file, 1, num_IRS);
    config.create_new_input_file = 0; % create done
    config.is_plot_graph = 0;
    [~, NN_1_res] = system([py_interpreter, ' ', py_UESENet_main_path, ...
                            ' --model ', 'vit', ...
                            ' --task ', 'cal', ...
                            ' --model_name ', 'vit_fold_9_test_3.pth', ...
                            ' --UE_list ', [num2str(num_UE_vec)], ...
                            ' --num_lab ', num2str(num_lab), ...
                            ' --max_num_sublab ', num2str(max_num_sublab)]);
    fprintf("[Info] numIRS=%d: NN-1 deduction done! Py Script cost time: %.4f s\n", num_IRS, cputime-nn_1_time);

    % NN-2 sample
    nn_2_time = cputime;
    gen_sys_sample(config, input_file, 2, num_IRS);
    [~, NN_2_res] = system([py_interpreter, ' ', py_UESENet_main_path, ...
                            ' --model ', 'cdf', ...
                            ' --task ', 'cal', ...
                            ' --model_name ', 'cdf.pth', ...
                            ' --UE_list ', [num2str(num_UE_vec)], ...
                            ' --num_lab ', num2str(num_lab), ...
                            ' --max_num_sublab ', num2str(max_num_sublab)]);
    fprintf("[Info] numIRS=%d: NN-2 deduction done! Py Script cost time: %.4f s\n", num_IRS, cputime-nn_2_time);

    %% create SE_matrix

    % dropout cur folder and create a new one
    if exist(save_data_path, 'dir')
        rmdir(save_data_path, 's');
    end
    mkdir(save_data_path);

    % calculate SE_matrix in each UE condition
    cal_SE_time = cputime;
    UE_sampleMatrix = load(input_file).UE_sampleMatrix;
    for ue_idx = 1:length(UE_sampleMatrix)
        num_UE = length(UE_sampleMatrix{ue_idx});
        gen_SE_time = gen_sys_SE_matrix(num_UE, num_served, config);
    end
    fprintf("[Info] numIRS=%d: SE_matrix generation done! Cal SE Cost time: %.4f s\n", num_IRS, cputime-cal_SE_time);

    %% Three Algorithms Comparison

    % generate cluster by KmeansConstr in python 3.10
    % cited by https://pypi.org/project/k-means-constrained/
    cal_kmeans_time = cputime;
    py_Kmeans_path = config.py_Kmeans_path;
    [~, cluster_res] = system([py_interpreter, ' ', py_Kmeans_path, ...
                                ' --ue_list ', [num2str(num_UE_vec)], ...
                                ' --folder_path ', save_data_path, ...
                                ' --num_slot ', num2str(num_slot), ...
                                ' --num_lab ', num2str(num_lab), ...
                                ' --max_num_sublab ', num2str(max_num_sublab)]);
    cluster_label = load(fullfile(save_data_path,'cluster_label.mat')).cluster_label;
    cluster_time = load(fullfile(save_data_path,'cluster_label.mat')).cluster_time;
    fprintf("[Info] numIRS=%d: Cluster label generation done! Cost time: %.4f s\n", num_IRS, cputime-cal_kmeans_time);

    % schedule resource
    cnt = 1;
    num_scheme = 3;
    num_UE_case = length(UE_sampleMatrix);
    
    % comparison between Gurobi/Heuristic/benchmarks
    Total_time = zeros(num_UE_case, num_scheme);
    Total_minSE = zeros(num_UE_case, num_scheme);

    % observation of IRS deploy param.
    SE_opt_cell = cell(num_UE_case, 1);

    for num_UE_idx = 1:num_UE_case

        num_UE = length(UE_sampleMatrix{num_UE_idx});
        SE_opt_matrix = [];

        for lab = 1:num_lab

            for sub_lab = 1:max_num_sublab

                % load SE_matrix
                SE_matrix_full_path = fullfile(save_data_path, ['SE_matrix_lab_', num2str(lab), '_', ...
                                                num2str(sub_lab), '_numUE_', num2str(num_UE), '.mat'] );
                if ~exist(SE_matrix_full_path, 'file')
                    break;
                end
                SE_tb_matrix = load(SE_matrix_full_path).SE_matrix;

                fprintf("[Info] numIRS=%d, numUE=%d: Lab_%d_sublab_%d is starting!\n", num_IRS, num_UE, lab, sub_lab);

                %% Algorithm 1: Gurobi Solver(Upper Bound)

                [Gurobi_minSE, Gurobi_time, opt_rho_alloc_Gurobi, opt_UE_IRS_matches_Gurobi] = GurobiUpperBound(num_UE, num_IRS, num_slot, SE_tb_matrix, save_data_path, print_opt_result);
                fprintf("[Info] numIRS=%d, numUE=%d: Gurobi global minSE: %.4f\n", num_IRS, num_UE, Gurobi_minSE);
                
                %% Algorithm 2: Heuristic Algorithm

                rand_heuristic_minSE = [];
                rand_Heuristic_time = [];
                num_rand_count = 1;
                for rand_cnt = 1:num_rand_count

                tic

                %%% Stage 0: Pre-allocation of UE in each slot %%%
                if ceil(num_UE / num_slot) ~= num_UE / num_slot
                    error("[Error] num_UE / num_slot must be integer");
                end
                numUE_perSlot = num_UE / num_slot; % assume always integer
                numUE_BS_served_perSlot = numUE_perSlot - num_IRS; % one IRS serves one UE, BS serves many UEs
                init_UE_alloc_matrix = Preprocess(config, SE_tb_matrix, numUE_perSlot, num_slot, cluster_label{cnt}); % dim: numSlot x numUE_perSlot
                init_rho_alloc_inSlot = repmat(1/numUE_perSlot, numUE_perSlot, 1); % dim: numUE_perSlot x 1

                %%% Stage 1: Local Optimization Problem (P2) Within a Time Slot %%%
                % illustrate the results of Stage 0
                % init_UE_alloc_matrix        % init_rho_alloc_inSlot
                %
                % SLOT       UE               % SLOT           RHO
                % slot1  UE3   UE1            % slot1  rho3(UE3)  rho1(UE1)
                % slot2  UE4   UE2            % slot2  rho4(UE4)  rho2(UE2)
                
                minSE_inSlot = zeros(num_slot, 1);
                opt_UE_IRS_matches = zeros(num_slot, numUE_perSlot);
                opt_rho_alloc_matrix = zeros(num_slot, numUE_perSlot);

                SE_tb_matrix_const = parallel.pool.Constant(SE_tb_matrix);
                for slot = 1:num_slot % can choose parfor if numSlot is huge

                    tmp_UE_group = init_UE_alloc_matrix(slot,:);
                    tmp_rho_alloc = init_rho_alloc_inSlot;
                    [cur_minSE, matches_result, tmp_rho_alloc] = max_minSE_inSlot(config, tmp_UE_group, tmp_rho_alloc, ...
                                                                    SE_tb_matrix_const, numUE_perSlot, num_IRS, num_served, num_UE, slot);

                    % ------- obtain the optimized result ------- %
                    minSE_inSlot(slot,1) = cur_minSE;
                    opt_UE_IRS_matches(slot,:) = matches_result;
                    opt_rho_alloc_matrix(slot,:) = tmp_rho_alloc;

                end % end for

                %%% Stage 2: optimization in whole resource %%%
                % illustrate the results of Stage 1
                % init_UE_alloc_matrix
                %
                % SLOT       UE
                % slot1  UE3   UE1
                % slot2  UE4   UE2
                %
                % opt_UE_IRS_matches
                %
                % SLOT                   PAIR
                % slot1   UE1-IRS1   UE2-   UE3-IRS2    UE4-
                % slot2     UE1-   UE2-IRS1    UE3-   UE4-IRS2
                %
                % opt_vec_rho_alloc               % minSE_inSlot
                %                                 % 
                % SLOT             RHO            % SLOT     minSE
                % slot1   rho3(UE3)   rho1(UE1)   % slot1     xxx
                % slot2   rho4(UE4)   rho2(UE2)   % slot2     xxx

                [init_UE_alloc_matrix, opt_rho_alloc_heuristic, opt_UE_IRS_matches_heuristic, minSE_inSlot] = ...
                                        max_minSE_acrossSlot(config, init_UE_alloc_matrix, opt_rho_alloc_matrix, opt_UE_IRS_matches, ...
                                                             SE_tb_matrix_const, minSE_inSlot, numUE_perSlot, num_IRS, num_UE);
                heuristic_minSE = min(minSE_inSlot);
                heuristic_time = toc;

                if print_opt_result == 1
                    print_optimized_result("Heuristic", "sub-OPTIMAL", heuristic_time, heuristic_minSE, ...
                                           init_UE_alloc_matrix, opt_rho_alloc_heuristic, opt_UE_IRS_matches_heuristic);
                end

                %% Benchmark: random allocate and match

                tic
                numUE_perSlot = num_UE / num_slot; % assume integer

                rand_id = randperm(num_UE);
                rand_UE_alloc_matrix = reshape(rand_id, num_slot, numUE_perSlot); % dim: numSlot x numUE_perSlot

                %%% original: random RB ratio allocation per slot
                % rand_rho = rand(num_slot, numUE_perSlot); % dim: num_slot x numUE_perSlot
                % bench_rho_alloc_random = rand_rho ./ sum(rand_rho, 2); % normalization

                %%% Revise version: equal RB ratio allocation per slot
                rho_alloc_random = 1/numUE_perSlot .* ones(1, numUE_perSlot);
                bench_rho_alloc_random = repmat(rho_alloc_random, num_slot, 1);

                bench_UE_IRS_matches_random = zeros(num_slot, numUE_perSlot);
                for slot = 1:num_slot
                    seq = randperm(numUE_perSlot);
                    rand_BS_serve_id = seq(1:numUE_BS_served_perSlot);
                    bench_UE_IRS_matches_random(slot,rand_BS_serve_id) = num_IRS + 1;
                    rand_IRS_serve_id = randperm(num_IRS);
                    bench_UE_IRS_matches_random(slot,seq(numUE_BS_served_perSlot+1:end)) = rand_IRS_serve_id;
                end

                bench_minSE = +inf;
                for slot = 1:num_slot
                    index = sub2ind(size(SE_tb_matrix), rand_UE_alloc_matrix(slot,:), bench_UE_IRS_matches_random(slot,:));
                    SE_inSlot = SE_tb_matrix(index) .* bench_rho_alloc_random(slot,:);
                    minSE = min(SE_inSlot);
                    bench_minSE = min(bench_minSE, minSE);
                end
                random_time = toc; % ms

                if print_opt_result == 1
                    print_optimized_result("Random", "Benchmark", random_time, ...
                        bench_minSE, rand_UE_alloc_matrix, bench_rho_alloc_random, bench_UE_IRS_matches_random)
                end

                rand_heuristic_minSE = [rand_heuristic_minSE heuristic_minSE];
                rand_Heuristic_time = [rand_Heuristic_time heuristic_time];
                end

                %% save SE_opt in all labs

                SE_opt_matrix = [SE_opt_matrix ; Gurobi_minSE heuristic_minSE bench_minSE];
                % SE_opt_matrix = [SE_opt_matrix ; heuristic_minSE bench_minSE];
                % SE_opt_matrix = [SE_opt_matrix ; mean(rand_heuristic_minSE) bench_minSE];
                cnt = cnt + 1;

            end % end sublab
        end % end lab

        SE_opt_cell{num_UE_idx} = SE_opt_matrix;

        Total_minSE(num_UE_idx, :) = [Gurobi_minSE heuristic_minSE bench_minSE];
        Total_time(num_UE_idx, :) = [Gurobi_time cluster_time(num_UE_idx)+heuristic_time random_time];
        % Total_minSE(num_UE_idx, :) = [heuristic_minSE bench_minSE];
        % Total_time(num_UE_idx, :) = [heuristic_time random_time];
        % Total_time(num_UE_idx, :) = [mean(rand_Heuristic_time) random_time];

    end % end num_ue_idx
    fprintf("[Info] numIRS=%d: Three Algorithms calculation done!\n", num_IRS_vec(num_IRS_idx));

    %% save data

    prefix = [];
    if config.use_bench_4_1_random == 1
        prefix = [prefix, 'bench_4_1_random_'];
    elseif config.use_bench_4_1_sort == 1
        prefix = [prefix, 'bench_4_1_sort_'];
    end

    if config.use_bench_4_2_enum == 1
        prefix = [prefix, 'bench_4_2_enum_'];
    elseif config.use_bench_4_2_greedy == 1
        prefix = [prefix, 'bench_4_2_greedy_'];
    elseif config.use_bench_4_2_Hungarian == 1
        prefix = [prefix, 'bench_4_2_Hungarian_'];
    end

    if config.use_bench_4_3_greedy == 1
        prefix = [prefix, 'bench_4_3_greedy_'];
    elseif config.use_bench_4_3_enhanced == 1
        prefix = [prefix, 'bench_4_3_enhanced_'];
    elseif config.use_4_3_enhanced == 1
        prefix = [prefix, '4_3_enhanced_'];
    end

    % output_file_path = fullfile(schedule_src_path, ...
    %     [prefix, 'SE_opt_cell_numIRS_', num2str(num_IRS), ...
    %     '_angle_', num2str(config.init_rot_angle), '.mat']);
    % save(output_file_path, "SE_opt_cell");

    output_sys_level_comp = fullfile(schedule_src_path, ...
        [prefix, 'SE_opt_Time_numIRS_', num2str(num_IRS), ...
        '_angle_', num2str(config.init_rot_angle), '.mat']);
    save(output_sys_level_comp, "Total_minSE", "Total_time");

end
fprintf("[Info] quit schedule.m. MATLAB total cost time: %.4f s\n", cputime-start_time);

% Turn off parallel pool
if config.pick_parfor == 1
    delete(gcp('nocreate'));
end

%% plot

% plot_result(config);
% fprintf("[Info] plot result done!\n");

end
