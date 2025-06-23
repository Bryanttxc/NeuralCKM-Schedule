% created by Bryanttxc, FUNLab

function SE_map(config)
%SE_MAP draw SE distribution map in multi-AIRS aided single-UE scene

% Include paths of other folders
root_path = config.root_path;
addpath(fullfile(root_path, 'util'));
addpath(fullfile(root_path, 'util', 'link_level'));

SE_map_src_folder = config.SE_map_src_path;
if ~exist(SE_map_src_folder, 'dir')
    mkdir(SE_map_src_folder);
end

% Turn on parallel pool
turnOnParfor(config);

%% Preparation

% lab param.
num_lab = config.num_lab;
max_num_sublab = config.max_num_sublab;
num_IRS_vec = config.num_IRS_vec;
num_BS = config.num_BS;
radius = config.radius;

default_create_file = config.create_new_input_file;
default_plot_graph = config.is_plot_graph;

%% Generate SE matrix

start_time = cputime;
for cur_radius = radius

    config.init_radius = cur_radius;

    for num_irs_idx = 1:length(num_IRS_vec)

        num_IRS = num_IRS_vec(num_irs_idx);
        num_served = num_IRS + num_BS;

        config.create_new_input_file = default_create_file; % reset
        config.is_plot_graph = default_plot_graph;

        LPS_Net_id = 1;
        SE_Net_id = 2;

        input_file = fullfile(SE_map_src_folder, ['link_level_data_numIRS_', num2str(num_IRS), '.mat']);
        py_interpreter = config.py_interpreter;
        py_uesenet_main_path = config.py_uesenet_main_path;

        % LPS-Net (NN-1)
        config = gen_link_sample(config, input_file, LPS_Net_id, num_IRS);
        UE_vec = config.numUE;
        tic
        [~, NN_1_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
                                ' --model ', 'vit', ...
                                ' --task ', 'cal', ...
                                ' --model_name ', 'vit_fold_9_test_3.pth', ...
                                ' --UE_list ', [num2str(UE_vec)], ...
                                ' --num_lab ', num2str(num_lab), ...
                                ' --max_num_sublab ', num2str(max_num_sublab)]);
        NN_time = toc;
        fprintf("[Info] numIRS=%d: LPS-Net deduction done! Cost time: %.4f s\n", num_IRS, NN_time);

        config.create_new_input_file = 0; % create done
        config.is_plot_graph = 0;

        % SE-Net (NN-2)
        config = gen_link_sample(config, input_file, SE_Net_id, num_IRS);
        tic
        [~, NN_2_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
                                ' --model ', 'cdf', ...
                                ' --task ', 'cal', ...
                                ' --model_name ', 'cdf.pth', ...
                                ' --UE_list ', [num2str(UE_vec)], ...
                                ' --num_lab ', num2str(num_lab), ...
                                ' --max_num_sublab ', num2str(max_num_sublab)]);
        NN_time = toc;
        fprintf("[Info] numIRS=%d: SE-Net deduction done! Cost time: %.4f s\n", num_IRS, NN_time);

        % calculate SE_matrix of each UE
        gen_SE_time = gen_link_SE_matrix(config, num_served);
        fprintf("[Info] numIRS=%d: SE_matrix generation done! Cost time: %.4f s\n", num_IRS, gen_SE_time);

    end % for num_irs_idx
end % for cur_radius 

end_time = cputime;
fprintf("Matlab total cost time: %.4f s\n", end_time - start_time);

%% Draw SE map

plot_SE_map(config);

end
