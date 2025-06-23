% created by Bryanttxc, FUNLab

function LoSLoS_scene(config)
% LOSLOS_SCENE compare the phase opt schemes at link level
% mission: LoS-LoS Scene Phase Optimization Comparison
% base on theoretical channel realization

%% load config param.

% include paths of other folders
root_path = config.root_path;
addpath(fullfile(root_path, 'algorithm'));
addpath(fullfile(root_path, 'util', 'link_level'));

save_data_path = config.save_data_path;
load_data_path = config.load_data_path;

% turn on parallel pool
turnOnParfor(config);

debug = config.debug;
scene = config.scene;
plot_powGain = config.plot_powGain;
save_dc_sca_data = config.save_dc_sca_data;

%% initialization

[General, Power, PathLoss, BS, AIRS, UE] = oneBS_oneIRS_oneUE_scene(config);

if scene == 1 % Scene 1: fixed position and calculate param.

    BS.x = 0;    BS.y = 0;
    AIRS.x = 80; AIRS.y = 0;
    UE.x = 80;   UE.y = 80;
    RotateMatrix = calRotateMatrix(AIRS.num, AIRS.horizon_rot_angle, zeros(AIRS.num,1), zeros(AIRS.num,1));
    [BS, AIRS, UE, Distance, Direction] = gen_layout(BS, AIRS, UE, RotateMatrix);

elseif scene == 2 % Scene 2: fixed param. directly

    % distance
    Distance.dist_BStoUE_3D = 100; % (m)
    Distance.dist_BStoAIRS_3D = 60;
    Distance.dist_AIRStoUE_3D = 120;
    
    % coordinate system with IRS as the origin
    Direction.theta_AIRStoBS_local = 144;
    Direction.phi_AIRStoBS_local = -45;
    Direction.theta_AIRStoUE_local = 120;
    Direction.phi_AIRStoUE_local = 5;
    
    Direction.direct_AIRStoBS_local = [sind(Direction.theta_AIRStoBS_local)*cosd(Direction.phi_AIRStoBS_local), ...
                                       sind(Direction.theta_AIRStoBS_local)*sind(Direction.phi_AIRStoBS_local), ...
                                       cosd(Direction.theta_AIRStoBS_local)];
    Direction.direct_AIRStoUE_local = [sind(Direction.theta_AIRStoUE_local)*cosd(Direction.phi_AIRStoUE_local), ...
                                       sind(Direction.theta_AIRStoUE_local)*sind(Direction.phi_AIRStoUE_local), ...
                                       cosd(Direction.theta_AIRStoUE_local)];
end

%% Subcarrier Increase

light_speed = General.light_speed;

M_Y_sets = General.M_Y_sets;
M_Z_sets = General.M_Z_sets;
num_sc_sets = General.num_sc_sets; % set of subcarriers

sumSE_MC = zeros(length(M_Y_sets), length(num_sc_sets));
sumSE_theory_approx = zeros(length(M_Y_sets), length(num_sc_sets));

for num_sc_idx = 1:length(num_sc_sets)

    % override args
    General.num_sc = num_sc_sets(num_sc_idx); % number of subcarriers

    freq_index = 1:General.num_sc;
    General.freq_set = General.freq_center + ...
                        General.bandwidth ./ General.num_sc .* (freq_index-1-(General.num_sc-1)./2);
    General.sc_interval = General.bandwidth ./ General.num_sc;
    
    Power.noise_power = Power.noise_density * General.sc_interval; % W
    Power.trans_power_per_sc = BS.trans_power ./ General.num_sc;
    Power.dynamicNoise_power = AIRS.dynamicNoise_power_density * General.sc_interval;
    
    %% channel
    
    [AIRS, PathLoss, Power] = gene_channel_para(BS, AIRS, PathLoss, Power, Direction, Distance);
    
    %% IRS Element Increase
    
    for num_elem_idx = 1:length(M_Y_sets)

        % override args
        AIRS.M_Y = M_Y_sets(num_elem_idx);
        AIRS.M_Z = M_Z_sets(num_elem_idx);
        AIRS.M = AIRS.M_Y .* AIRS.M_Z;
        AIRS.element_y_indices = 0:AIRS.M_Y-1;
        AIRS.element_z_indices = 0:AIRS.M_Z-1;

        fprintf("\n[Info] Number of Subcarrier: %d, Number of AIRS Element: %d x %d is running!\n", ...
                    General.num_sc, AIRS.M_Y, AIRS.M_Z);
        
        %% Array response vector (ARV)
        
        num_sc = General.num_sc;
        freq_set = General.freq_set;
        RotateMatrix = calRotateMatrix(AIRS.num, AIRS.horizon_rot_angle, zeros(AIRS.num,1), zeros(AIRS.num,1));
        [ARV_incident, ARV_reflect, ARV_y, ARV_z] = gen_ARV(AIRS, Direction, light_speed, num_sc, freq_set, RotateMatrix);

        %% Phase Optimization
        
        sqrt_ERP_AIRStoUE = diag(sqrt(Power.ERP_AIRStoUE));
        CCM = zeros(AIRS.M, AIRS.M, num_sc);
        for sc = 1:num_sc
            ARV_reflect_sc = ARV_reflect(:,sc);
            ARV_incident_sc = ARV_incident(:,sc);
            CCM(:,:,sc) = diag(ARV_incident_sc')*conj(ARV_reflect_sc)...
                            *(sqrt_ERP_AIRStoUE' * Power.ERP_BStoAIRS * sqrt_ERP_AIRStoUE)*...
                          (ARV_reflect_sc.')*diag(ARV_incident_sc);
        end
        
        % debugger
        % dc_sca version
        if debug == 1
            ARV_yz = zeros(M,num_sc);
            CCM_test = zeros(M,M,num_sc);
            for sc = 1:num_sc
                ARV_yz(:,sc) = kron(ARV_y(:,sc) , ARV_z(:,sc) );
                CCM_test(:,:,sc) = conj(ARV_yz(:,sc)) * ARV_yz(:,sc).' * Power.ERP_AIRStoUE * Power.ERP_BStoAIRS;
                diff_ccm = CCM_test(:,:,sc) - CCM(:,:,sc); % e-20 -> equal
            end
        end
        
        %--------------------- METHOD1: MCCM (对CCM做平均)---------------------%
        MCCM = mean(CCM,3);
        [U_mccm,S_mccm,~] = svd(MCCM); % check S_mccm if it is dominated
        phi_MCCM_SVD = exp(1j*angle(U_mccm(:,1))); % 取负号与phi_align_fc一样
        
        %--------------------- METHOD2: Align center freq (单频点单位置)---------------------%
        freq_center_index = round(num_sc/2);
        phi_align_fc = exp(-1j*( angle(ARV_reflect(:,freq_center_index)) + angle(ARV_incident(:,freq_center_index)) ));
        
        if debug == 1
            powGain_debugger(CCM, ARV_y, ARV_z, AIRS.M_Y, AIRS.M_Z, Power.ERP_BStoAIRS, Power.ERP_AIRStoUE); % validation
        end
        
        %--------------------- METHOD3: No-beamforming (随机相位) ---------------------%
        phi_random = exp(1j*2*pi*rand(AIRS.M,1));
        
        % %--------------------- METHOD4: DC-SCA algorithm (宽带单位置) ---------------------%
        % path = ['LoS_LoS_phi_dc_sca_',num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z), ...
        %       '_BW_',num2str(General.bandwidth/1e6), ...
        %       'MHz_fc_',num2str(General.freq_center/1e9), 'GHz','.mat'];
        % if save_dc_sca_data == 1
        %     phi_fc_subopt_y = ARV_y(:,freq_center_index);
        %     phi_fc_subopt_z = ARV_z(:,freq_center_index);
        %     [phi_dc_sca, Phi_y_opt, Phi_z_opt] = AO_dc_sca_algorithm(ARV_y, ARV_z, phi_fc_subopt_y, phi_fc_subopt_z, Power.ERP_AIRStoUE*Power.ERP_BStoAIRS, num_sc);
        %     save(strcat(save_data_path,path),"phi_dc_sca");
        % else
        %    phi_dc_sca = cell2mat(struct2cell(load(fullfile(link_data_path, path))));
        % end
        
        %% Power Gain indicator
        
        powGain_dB_mccm = zeros(num_sc,1);
        powGain_dB_afc = zeros(num_sc,1);
        powGain_dB_rand = zeros(num_sc,1);
        % powGain_dB_dc_sca = zeros(num_sc,1);
        for sc = 1:num_sc
            powGain_dB_mccm(sc) = 10*log10(real(trace( CCM(:,:,sc) * (phi_MCCM_SVD * phi_MCCM_SVD') )));
            
            powGain_dB_afc(sc) = 10*log10(real(trace( CCM(:,:,sc) * (phi_align_fc * phi_align_fc') )));
            
            powGain_dB_rand(sc) = 10*log10(real(trace( CCM(:,:,sc) * (phi_random * phi_random') )));
            
            % powGain_dB_dc_sca(sc) = 10*log10(real(trace( CCM(:,:,sc) * (phi_dc_sca * phi_dc_sca') )));
        end
        
        %% Plot
        
        if plot_powGain == 1

            freq_GHz = freq_set ./ 1e9;
            linewidth = config.linewidth;
            linestyle = config.linestyle;
            color = config.color;
            
            figure;
            set(gcf,'color','w');
            plot(freq_GHz', powGain_dB_mccm, 'LineStyle', linestyle, 'color', color(1), 'linewidth', linewidth);
            hold on
            plot(freq_GHz', powGain_dB_afc, 'LineStyle', linestyle, 'color', color(2), 'linewidth', linewidth);
            plot(freq_GHz', powGain_dB_rand, 'LineStyle', linestyle, 'color', color(3), 'linewidth', linewidth);
            % plot(freq_GHz', powGain_dB_dc_sca, 'color', color(4), 'linewidth', linewidth);
            legend("MCCM-SVD", "Align Fc", "Random", 'location', 'best');
            if General.bandwidth/1e9 <= 1
                titlestr = [num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z),',BW: ', num2str(General.bandwidth/1e6),'MHz'];
            else
                titlestr = [num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z),',BW: ', num2str(General.bandwidth/1e9),'GHz'];
            end
            title(titlestr);
            xlabel("Frequency(GHz)");
            ylabel("Power Gain(dB)");
            set(gca, 'fontsize', config.fontsize);
        end
        
        %% theory formula
        
        % 子载波数较少的时候，误差较大
        phi = phi_MCCM_SVD;
        
        % direct link
        powGain_direct = PathLoss.PL_BStoUE;
        
        % cascaded link
        const_term = AIRS.M^2 * AIRS.amplify_factor^2 * PathLoss.PL_BStoAIRS * PathLoss.PL_AIRStoUE;
        channel_gain = zeros(num_sc,1);
        for sc = 1:num_sc
            channel_gain(sc) = abs(phi' * CCM(:,:,sc) * phi);
        end
        avg_term = mean(channel_gain);
        powGain_cascade = const_term * avg_term;
        
        signalPart = Power.trans_power_per_sc * (powGain_direct + powGain_cascade);
        noisePart = AIRS.M * PathLoss.PL_AIRStoUE * Power.ERP_AIRStoUE * AIRS.amplify_factor^2 * Power.dynamicNoise_power + Power.noise_power;
        sumSE_theory_approx(num_elem_idx, num_sc_idx) = num_sc * log2(1 + signalPart / noisePart);
        
        %% MC simulation
        
        num_fading = config.num_small_fading;
        funcDelay = General.funcDelay;
        
        SE_MC_MCCM_SVD = zeros(num_sc,num_fading);
        SE_MC_align_fc = zeros(num_sc,num_fading);
        SE_MC_random = zeros(num_sc,num_fading);
        % R_MC_dc_sca = zeros(num_sc,num_fading);
        
        Phi_MCCM_SVD = diag(phi_MCCM_SVD);
        Phi_align_fc = diag(phi_align_fc);
        Phi_random = diag(phi_random);
        % Phi_dc_sca = diag(phi_dc_sca);
        
        tau_BStoUE = Distance.dist_BStoUE_3D / General.light_speed;
        tau_BStoAIRS = Distance.dist_BStoAIRS_3D / General.light_speed;
        tau_AIRStoUE = Distance.dist_AIRStoUE_3D / General.light_speed;
        
        tic
        fprintf("[Info] MC simulation start...\n")
        %------------------ Ergodic SE ------------------%
        for numf = 1:num_fading % if channel_BStoUE is NLoS
            
            for sc = 1:num_sc
        
                h_BStoUE_sc = sqrt(PathLoss.PL_BStoUE).*funcDelay(freq_set(sc),tau_BStoUE); % multipath
                h_BStoAIRS_sc = sqrt(AIRS.M) * sqrt(PathLoss.PL_BStoAIRS)*funcDelay(freq_set(sc),tau_BStoAIRS)*sqrt(Power.ERP_BStoAIRS) * ARV_incident(:,sc); % M x 1
                h_AIRStoUE_sc = sqrt(AIRS.M) * sqrt(PathLoss.PL_AIRStoUE)*funcDelay(freq_set(sc),tau_AIRStoUE)*sqrt(Power.ERP_AIRStoUE) * ARV_reflect(:,sc); % M x 1
                
                func_signalPow = @(Phi) Power.trans_power_per_sc * abs(h_BStoUE_sc + h_AIRStoUE_sc.' * AIRS.amplify_factor * Phi * h_BStoAIRS_sc)^2;
                func_noisePow = @(Phi) norm(h_AIRStoUE_sc.' * Phi)^2 * AIRS.amplify_factor^2* Power.dynamicNoise_power + Power.noise_power;
        
                %----------------------- MCCM (江涛 对CCM做平均) -----------------------%
                SE_MC_MCCM_SVD(sc,numf) = log2(1 + func_signalPow(Phi_MCCM_SVD) / func_noisePow(Phi_MCCM_SVD));
                
                %----------------------- Align center freq (单频点单位置) -----------------------%
                SE_MC_align_fc(sc,numf) = log2(1 + func_signalPow(Phi_align_fc) / func_noisePow(Phi_align_fc));
        
                %----------------------- random phase (随机相位) -----------------------%
                SE_MC_random(sc,numf) = log2(1 + func_signalPow(Phi_random) / func_noisePow(Phi_random));
        
                % %----------------------- DC-SCA algorithm (宽带单位置) -----------------------%
                % SE_MC_dc_sca(sc,numf) = log2(1 + func_signalPow(Phi_dc_sca) / func_noisePow(Phi_dc_sca));
        
            end
        end
        ergodic_SE_MC_mccm   = mean(SE_MC_MCCM_SVD,2);
        ergodic_SE_MC_afc    = mean(SE_MC_align_fc,2);
        ergodic_SE_MC_random = mean(SE_MC_random,2);
        % ergodic_SE_MC_dc_sca = mean(R_MC_dc_sca,2);
        time_t = toc;
        fprintf("[Info] MC simulation finish! Cost time: %.4f s\n", time_t);
        
        %% validate theory formula
        
        sumSE_MC(num_elem_idx,num_sc_idx) = sum(ergodic_SE_MC_mccm);
        diff = abs(sumSE_MC(num_elem_idx,num_sc_idx) - sumSE_theory_approx(num_elem_idx,num_sc_idx));
        fprintf("[Info] Diff between MC and theory formula is %.4f bps/Hz\n", diff);
        pause(1);
        
    end % end element increment

end % end sc increment

%% Save data

path = ['LoS_LoS_AIRS_elem_from_',num2str(M_Y_sets(1)), 'x', num2str(M_Y_sets(1)),'to', ...
        num2str(M_Y_sets(end)), 'x', num2str(M_Y_sets(end)), ...
        '_SC_from_', num2str(num_sc_sets(1)), 'to', num2str(num_sc_sets(end)), ...
        '_BW_',num2str(General.bandwidth/1e6),'MHz_fc_',num2str(General.freq_center/1e9),'GHz','.mat'];
sumSE_matrix{1} = sumSE_MC;
sumSE_matrix{2} = sumSE_theory_approx;
save(fullfile(save_data_path, path),'sumSE_matrix');

%% Plot

freq_GHz = freq_set./1e9;
linewidth = config.linewidth;
fontname = config.fontname;
color = config.color;
fontsize = config.fontsize;

figure;
set(gcf,'color','w');
plot(freq_GHz', ergodic_SE_MC_mccm, 'color', color(1), 'linewidth', linewidth);
hold on
plot(freq_GHz', ergodic_SE_MC_afc, 'color', color(2), 'linewidth', linewidth);
plot(freq_GHz', SE_MC_random, 'color', color(3), 'linewidth', linewidth);
% plot(freq_GHz', R_MC_dc_sca, 'color', color(4), 'linewidth', linewidth);
legend("MCCM-SVD", "Align Fc", "Random", 'location', 'best', 'Fontname', fontname);
if General.bandwidth/1e9 <= 1
    titlestr = ['AIRS: ',num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z),',BW: ', num2str(General.bandwidth/1e6),'MHz'];
else
    titlestr = ['AIRS: ',num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z),',BW: ', num2str(General.bandwidth/1e9),'GHz'];
end
title(titlestr, 'FontName', fontname);
xlabel("Frequency(GHz)", 'FontName', fontname);
ylabel("Spectral Efficiency (bps/Hz)", 'FontName', fontname);
set(gca, 'fontsize', fontsize);
