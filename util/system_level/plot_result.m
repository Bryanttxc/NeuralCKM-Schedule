% created by Bryanttxc, FUNLab

function plot_result(config)

close all

%% initialization

schedule_src_path = config.schedule_src_path;

linewidth = config.linewidth;
linestyle = config.linestyle;
% fontsize = config.fontsize;
fontsize = 30;
fontname = config.fontname;
color = config.three_color;

num_IRS_vec = config.num_IRS_vec;
num_IRS_vec = 2:4:6;
numIRS_len = length(num_IRS_vec);

angle = 90:30:180;

output_SE_cell = cell(numIRS_len, 4);
for numIRS_idx = 1:numIRS_len
    for angle_id = 1:4
        file_name = fullfile(schedule_src_path, ...
            ['SE_opt_cell_numIRS_', num2str(num_IRS_vec(numIRS_idx)), ...
            '_angle_', num2str(angle(angle_id)), '.mat']);
        output_SE_cell{numIRS_idx, angle_id} = load(file_name).SE_opt_cell;
    end
end

%% observation

UE_vec = config.UE_vec; % UE数的case
numUE_len = length(UE_vec);

%% observation 1 -- 单元数
% x-axis: num_UE
% y-axis: SE

% lab 1
element_vec = config.element_vec;
% lab 2
dist_vec = config.horizon_dist_vec;
% lab 3
height_vec = config.height_vec;
% lab 4
rotate_vec = config.rotate_vec;
% lab 5
Gi_vec = config.Gi_vec;

% LAB{1} = element_vec.^2;
% LAB{2} = dist_vec;
% LAB{3} = height_vec;
% LAB{4} = rotate_vec;
% LAB{5} = Gi_vec;
LAB{1} = dist_vec;

endIdx = 1;

marker = ["square", "o", "x", "diamond"];
linewidth = 5;
linestyle = ["--","-"];
color = [253,184,99;
    230,97,1;
    178,171,210;
    94,60,153] ./ 255;

for lab = 1:length(LAB)

    lab_vec = LAB{lab};
    num_case = length(lab_vec);
    
    figure; % 'position',[350,200,800,550]
    set(gcf, 'color', 'w');

    for numUE_idx = numUE_len% UE数
        
        legend_name = [];
        for angle_id = 1:4

            tmp_numIRS = [];     
            for numIRS_idx = 1:numIRS_len % IRS面板数
                sel_numIRS_cell = output_SE_cell{numIRS_idx, angle_id}; % e.x. 2-IRS
                
                tmp = [];
                for case_idx = 1:num_case % case数
                    tmp = [tmp; sel_numIRS_cell{numUE_idx}(endIdx+case_idx-1,:)];
                end
                tmp_numIRS = [tmp_numIRS; tmp];
            end
    
            % % create subplot -- e.x. IRS element: 8x8
            % num_row = floor(sqrt(num_case));
            % num_col = ceil(num_case / num_row);
            % subplot(num_row, num_col, case_idx);
    
            SE_vec = reshape(tmp_numIRS(:,1), num_case, []);
            plot(lab_vec, SE_vec(:,1)', 'Color', color(angle_id,:), 'LineStyle', linestyle(1), 'LineWidth', linewidth, 'Marker', marker(angle_id), 'MarkerSize', 20, 'MarkerFaceColor','none', 'MarkerEdgeColor','auto')
            hold on
            plot(lab_vec, SE_vec(:,2)', 'Color', color(angle_id,:), 'LineStyle', linestyle(2), 'LineWidth', linewidth, 'Marker', marker(angle_id), 'MarkerSize', 20, 'MarkerFaceColor','none', 'MarkerEdgeColor','auto')
            % plot(lab_vec, SE_vec(:,3)', 'Color', 'b', 'LineStyle', linestyle, 'LineWidth', linewidth, 'Marker','square', 'MarkerFaceColor','none', 'MarkerEdgeColor','auto')
    
            % h = bar(UE_vec, SE_vec, 1);
            % set(gca,'XTickLabel',num2str(UE_vec),'FontSize',fontsize,'FontName',fontname);
            % for numIRS_idx = 1:numIRS_len
            %     set(h(numIRS_idx),'FaceColor',color(numIRS_idx,:));  
            % end
            % box on
    
            % grid on
            % set(gca, 'GridLineWidth', 0.5);
            legend_name = [legend_name; num_IRS_vec' + "-IRS-Angle-" + angle(angle_id)];
    
            % if lab == 1
            %     xlabel('IRS Element', 'FontName', fontname);
            % elseif lab == 2
            %     xlabel('Horizontal Distance(m)', 'FontName', fontname);
            % elseif lab == 3
            %     xlabel('Height(m)', 'FontName', fontname);
            % elseif lab == 4
            %     xlabel('Rotate Angle($^{\circ}$)', 'Interpreter','latex', 'FontName', fontname);
            % elseif lab == 5
            %     xlabel('max ERP(dBi)', 'FontName', fontname);
            % end

        end

        legend(legend_name,'Location','best', 'FontName', fontname);
        xlabel('Horizontal Distance(m)', 'FontName', fontname);
        ylabel('Minimum Ergodic SE(bps/Hz)', 'Fontname', fontname);
        % title(['Rotate Angle: ', num2str(config.init_rot_angle), '$^{\circ}$'], 'interpreter', 'latex', 'FontName', fontname);
        set(gca, 'fontsize', fontsize);
    
    end
    endIdx = endIdx + num_case;

end

end
