% created by Bryanttxc, FUNLab

function plot_SE_map(config)
% PLOT_SE_MAP plot SE map

close all

save_data_path = config.save_data_path;
SE_map_src_path = config.SE_map_src_path;

radius = config.radius;
num_UE = config.num_UE;
num_IRS_vec = 6;
lab = 1;
sub_lab = 1;

fontsize = config.fontsize;
fontname = config.fontname;

for num_irs_idx = 1:length(num_IRS_vec)

    num_IRS = num_IRS_vec(num_irs_idx);

    fprintf("[Info] numIRS=%d, numUE=%d: plot Lab_%d_sublab_%d is starting!\n", ...
            num_IRS, num_UE, lab, sub_lab);
    
    fig = figure('Name', "numIRS_" + num2str(num_IRS));
    set(gcf,'Color','w');

    num_rad = length(radius);
    num_row = floor(sqrt(num_rad));
    num_col = ceil(num_rad / num_row);
    cnt = 1;

    for cur_rad = radius

        % input_file = fullfile(SE_map_src_path, ['link_level_data_numIRS_', num2str(num_IRS), '.mat']);
        % UE_sampleMatrix = load(input_file).UE_sampleMatrix;
        % num_UE = length(UE_sampleMatrix);

        % load SE_matrix
        SE_matrix_full_path = fullfile(save_data_path, ['SE_matrix_lab_', num2str(lab), '_', num2str(sub_lab), ...
                              '_numUE_', num2str(num_UE), '_numIRS_', num2str(num_IRS), '_', num2str(cur_rad), '.mat'] );
        if ~exist(SE_matrix_full_path, 'file')
            return;
        end
        SE_tb_matrix = load(SE_matrix_full_path).SE_matrix;

        subplot(num_row, num_col, cnt)

        % for idx = 1:size(SE_tb_matrix,2)-1

            % [max_SE_tb, max_idx] = max(SE_tb_matrix, [], 2);
            tmp_SE_matrix = SE_tb_matrix(:,1:end-1);
            max_SE_tb = max(tmp_SE_matrix,[],2) - SE_tb_matrix(:,end); % IRS_GAIN
            % max_SE_tb = max(tmp_SE_matrix,[],2);

            Xmesh3D = unique(UE_sampleMatrix(:,1));
            Ymesh3D = unique(UE_sampleMatrix(:,2));
            max_SE_tb = reshape(max_SE_tb, length(unique(Xmesh3D)), []);

            % subplot(num_row, num_col, idx);
            % set(h1,'position',[0.07,0.58,0.38,0.38])
            imagesc(Xmesh3D, Ymesh3D, max_SE_tb);
            % slice(Xmesh3D,Ymesh3D,max_SE_tb);
            % axis([min(tmp_ue_x),max(tmp_ue_x),min(tmp_ue_y),max(tmp_ue_y)]);
            axis xy
            grid on;
            colormap("jet");
            hcolor=colorbar;
            clim([0,2.5]);
            set(get(hcolor,'Title'),'string','bps/Hz','FontSize',20);
            set(gca,'FontSize',fontsize);
            shading interp % remove the color of frame
            xlabel("$x$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);
            ylabel("$y$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);
            titlestr = ['d=', num2str(cur_rad), ' m'];
            % zlabel("$z$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);

            % if idx == size(SE_tb_matrix,2)
            %     titlestr = 'BS-served';
            % else
            %     titlestr = ['IRS', num2str(idx), '-served-Gain'];
            % end
            title(titlestr, 'FontSize', fontsize, 'FontName', fontname);
        % end
        cnt = cnt + 1;
        % save_path = fullfile(config.save_fig_path, 'hor_dist_IRS_gain_change', ...
        %                     ['tmp_tmp_numIRS_', num2str(num_IRS), '_', num2str(cur_radius), '.fig']);
        % saveas(fig, save_path);
    end
end

num_IRS_1 = 2;
num_IRS_2 = 6;

input_file = fullfile(SE_map_src_path, ['link_level_data_numIRS_', num2str(num_IRS_1), '.mat']);
UE_sampleMatrix = load(input_file).UE_sampleMatrix;
num_UE = length(UE_sampleMatrix);

for cur_rad = radius

    % load SE_matrix
    SE_matrix_full_path_1 = fullfile(save_data_path, ['SE_matrix_lab_', num2str(lab), '_', num2str(sub_lab), ...
                            '_numUE_', num2str(num_UE), '_numIRS_', num2str(num_IRS_1), '_', num2str(cur_rad), '.mat'] );
    if ~exist(SE_matrix_full_path_1, 'file')
        return;
    end
    SE_tb_matrix_1 = load(SE_matrix_full_path_1).SE_matrix;

    SE_matrix_full_path_2 = fullfile(save_data_path, ['SE_matrix_lab_', num2str(lab), '_', num2str(sub_lab), ...
                            '_numUE_', num2str(num_UE), '_numIRS_', num2str(num_IRS_2), '_', num2str(cur_rad), '.mat'] );
    if ~exist(SE_matrix_full_path_2, 'file')
        return;
    end
    SE_tb_matrix_2 = load(SE_matrix_full_path_2).SE_matrix;

    fig = figure('Name', "horizon_dist_" + num2str(cur_rad));
    set(gcf,'Color','w');

    max_SE_tb = zeros(size(SE_tb_matrix_1,1), 2);

    % [max_SE_tb, max_idx] = max(SE_tb_matrix, [], 2);
    tmp_SE_matrix_1 = SE_tb_matrix_1(:,end);
    max_SE_tb(:,1) = max(tmp_SE_matrix_1,[],2);

    tmp_SE_matrix_2 = SE_tb_matrix_2(:,end);
    max_SE_tb(:,2) = max(tmp_SE_matrix_2,[],2);

    diff_SE_tb = max_SE_tb(:,1) - max_SE_tb(:,2);
    BetterIS2 = diff_SE_tb > 0;
    num_region_IRS2 = length(find(BetterIS2 == 1));
    num_region_IRS6 = length(find(BetterIS2 == 0));
    fprintf("[Info] Number of better regions for IRS2: %d, Number of better regions for IRS6: %d\n", ...
        num_region_IRS2, num_region_IRS6);

    Xmesh3D = unique(UE_sampleMatrix(:,1));
    Ymesh3D = unique(UE_sampleMatrix(:,2));
    diff_SE_tb = reshape(diff_SE_tb, length(unique(Xmesh3D)), []);

    max_SE_tb_1 = reshape(max_SE_tb(:,1), length(unique(Xmesh3D)), []);
    max_SE_tb_2 = reshape(max_SE_tb(:,2), length(unique(Xmesh3D)), []);

    % subplot(num_row, num_col, idx);
    % set(h1,'position',[0.07,0.58,0.38,0.38])
    imagesc(Xmesh3D, Ymesh3D, diff_SE_tb);
    axis xy
    grid on;
    colormap("jet");
    hcolor=colorbar;
    clim([0,2]);
    set(get(hcolor,'Title'),'string','bps/Hz','FontSize',20);
    set(gca,'FontSize',fontsize);
    shading interp % remove the color of frame
    xlabel("$x$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);
    ylabel("$y$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);

    % figure;
    % imagesc(Xmesh3D, Ymesh3D, max_SE_tb_1);
    % axis xy
    % grid on;
    % colormap("jet");
    % hcolor=colorbar;
    % clim([0,2]);
    % set(get(hcolor,'Title'),'string','bps/Hz','FontSize',20);
    % set(gca,'FontSize',fontsize);
    % shading interp % remove the color of frame
    % xlabel("$x$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);
    % ylabel("$y$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);
    % 
    % figure;
    % imagesc(Xmesh3D, Ymesh3D, max_SE_tb_2);
    % % slice(Xmesh3D,Ymesh3D,max_SE_tb);
    % % axis([min(tmp_ue_x),max(tmp_ue_x),min(tmp_ue_y),max(tmp_ue_y)]);
    % axis xy
    % grid on;
    % colormap("jet");
    % hcolor=colorbar;
    % clim([0,2]);
    % set(get(hcolor,'Title'),'string','bps/Hz','FontSize',20);
    % set(gca,'FontSize',fontsize);
    % shading interp % remove the color of frame
    % xlabel("$x$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);
    % ylabel("$y$(m)",'interpreter','latex','FontSize',fontsize, 'FontName', fontname);

    % if idx == size(SE_tb_matrix,2)
    %     titlestr = 'BS-served';
    % else
    %     titlestr = ['IRS', num2str(idx), '-served-Gain'];
    % end
    % title(titlestr, 'FontSize', 20, 'FontName', fontname);
end

end
