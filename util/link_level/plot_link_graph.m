% created by Bryanttxc, FUNLab

function plot_link_graph(input_file, num_IRS, root_path)
%PLOT_GRAPH plot simulation scene
%
% %%% input %%%
% input_file: input data file
% root_path: root folder

addpath(fullfile(root_path, 'util', 'system_level'));
addpath(fullfile(root_path, 'channel', 'quadriga_src'));
addpath(fullfile(root_path, 'channel', 'qris-tidy', 'class'));
addpath(fullfile(root_path, 'channel', 'qris-tidy', 'func'));

SAMPLE = load(input_file);
General = SAMPLE.General;
UE_sampleMatrix = SAMPLE.UE_sampleMatrix;
BS = SAMPLE.BS;
AIRS_set = SAMPLE.AIRS_set;
UE = SAMPLE.UE;

smlt_para = qd_simulation_parameters; % simulation parameters
smlt_para.center_frequency = General.freq_center;
smlt_para.sample_density = 4; % 4 samples per half-wavelength, relevant to Nyquist
smlt_para.use_absolute_delays = 1; % Include delay of the LOS path
smlt_para.show_progress_bars = 0; % Disable progress bars
smlt_para.use_3GPP_baseline = 1; % Enable drifting, default = 0

scenario_name_LOS = '3GPP_38.901_UMa_LOS';
scenario_name_NLOS = '3GPP_38.901_UMa_NLOS';

BS_Position = [BS.x BS.y BS.z];
BS.Antennas = 1;
Tx = class_Tx_linear(BS_Position, BS.Antennas, smlt_para.wavelength, 'omni', smlt_para.center_frequency); 

AIRS_COS = cell(num_IRS,1);
cnt = 1;
for irs_idx = 1:num_IRS
    AIRS = AIRS_set{irs_idx};
    AIRS_position = [AIRS.x, AIRS.y, AIRS.z];
    AIRS_COS{cnt} = class_RIS_singlepol(AIRS_position, AIRS.M_Y, AIRS.M_Z, ...
                                             General.wavelength_center, General.freq_center, ...
                                             AIRS.antenna_gain, AIRS.q, ...
                                             AIRS.xrot, AIRS.yrot, AIRS.zrot); 
    cnt = cnt + 1;
end

for num_UE_idx = 1

    tmp_UE_sampleMatrix = UE_sampleMatrix;

    numUE = size(tmp_UE_sampleMatrix, 1);
    UE_Position = [tmp_UE_sampleMatrix(:,1:2), UE.z*ones(numUE, 1)];
    Rx = cell(1, 1);
    for ue_idx = 1
        Rx{ue_idx} = class_Rx_linear(UE_Position(ue_idx,:), UE.Antennas, smlt_para.wavelength, 'omni', smlt_para.center_frequency);
    end

    l = func_get_QuaDRiGa_layout(smlt_para, {Tx}, Rx, AIRS_COS, scenario_name_LOS, scenario_name_NLOS); % QRIS author created
    h = l.visualize(1:l.no_tx, 1:l.no_rx, 0);
    
    hold on
    
    L_line = 25; % length of line
    L_arrow = 10; % length of arrow
    color = [255 128 0]./255;
    linewidth = 3;
    arrow_head_size = 4;
    arrow_angle_offset = 30;
    
    for irs_idx = 1:num_IRS

        AIRS = AIRS_set{irs_idx};

        % start pos & end pos of line
        x0 = AIRS.x;
        y0 = AIRS.y;
        x1 = x0;
        x2 = x0;
        y1 = y0 - L_line/2;
        y2 = y0 + L_line/2;
        
        % rotate matrix（deg->rad）
        rot_angle = AIRS.zrot;
        rot_angle = deg2rad(rot_angle);
        R = [cos(rot_angle), -sin(rot_angle); sin(rot_angle), cos(rot_angle)];
        
        % rotate line
        P1 = R * [x1 - x0; y1 - y0] + [x0; y0];
        P2 = R * [x2 - x0; y2 - y0] + [x0; y0];
        plot([P1(1), P2(1)], [P1(2), P2(2)], 'Color', color, 'LineStyle', '--', 'LineWidth', linewidth, 'HandleVisibility', 'off');
        
        % length & rotate angle of arrow
        tail = AIRS.boresight; % 旋转角度（度）
        x1 = L_arrow * tail(1) + x0;
        y1 = L_arrow * tail(2) + y0;
        
        rot_angle = AIRS.zrot;
        dx = L_arrow * cosd(rot_angle);
        dy = L_arrow * sind(rot_angle);
        plot([x0, x1], [y0, y1], 'Color', color, 'LineStyle', '-', 'LineWidth', linewidth, 'HandleVisibility', 'off');
        arrow_head_x = [x1, x1 - arrow_head_size*cosd(rot_angle + arrow_angle_offset), x1 - arrow_head_size*cosd(rot_angle - arrow_angle_offset)];
        arrow_head_y = [y1, y1 - arrow_head_size*sind(rot_angle + arrow_angle_offset), y1 - arrow_head_size*sind(rot_angle - arrow_angle_offset)];
        
        % head of arrow
        fill(arrow_head_x, arrow_head_y, color, 'EdgeColor', color, 'MarkerSize', 10, 'HandleVisibility', 'off');
        set(gca, 'fontsize', 28);
    end

    set(h,'PaperSize',[23.5 17]);
    set(gca,'fontsize',28);
end

end
