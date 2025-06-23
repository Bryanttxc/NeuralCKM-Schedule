% created by Bryanttxc, FUNLab

function LoSNLoS_scene(config)
% LOSNLOS_SCENE compare the phase opt schemes at link level
% mission: LoS-NLoS Scene Phase Optimization Comparison
% base on theoretical channel realization

%% load config param.

% Include paths of other folders
root_path = config.root_path;
addpath(fullfile(root_path, 'algorithm'));
addpath(fullfile(root_path, 'util', 'link_level'));

save_data_path = config.save_data_path;
load_data_path = config.load_data_path;

% turn on parallel pool
turnOnParfor(config);

num_angleTrial = config.num_angleTrial;
num_fading = config.num_small_fading;
rand_seed = config.rand_seed;

save_rcg_data = config.save_rcg_data;
debug = config.debug;

%% initialization

[General, Power, PathLoss, BS, AIRS, UE] = oneBS_oneIRS_oneUE_scene(config);

% position
BS.x = 0;    BS.y = 0;
AIRS.x = 80; AIRS.y = 0;
UE.x = 80;   UE.y = 80;
RotateMatrix = calRotateMatrix(AIRS.num, AIRS.horizon_rot_angle, zeros(AIRS.num,1), zeros(AIRS.num,1));
[BS, AIRS, UE, Distance, Direction] = gen_layout(BS, AIRS, UE, RotateMatrix);

%------------------ multipath direction ------------------%
[Direction.theta_AIRStoUE_path_local,Direction.phi_AIRStoUE_path_local] ...
        = gene_multipath_angle(AIRS.horizon_rot_angle, General.num_path_AIRStoUE, num_angleTrial, 'uniform', rand_seed); % Lr x 3 x num_angleTrial

%% Subcarrier Increase

M_Y_sets = General.M_Y_sets;
M_Z_sets = General.M_Z_sets;
num_sc_sets = General.num_sc_sets; % subcarriers

avg_sumSE_MC = zeros(length(M_Y_sets), length(num_sc_sets));
avg_sumSE_theory_approx = zeros(length(M_Y_sets), length(num_sc_sets));

for num_sc_idx = 1:length(num_sc_sets)

    % override args
    General.num_sc = num_sc_sets(num_sc_idx); % subcarriers

    freq_index = 1:General.num_sc;
    General.freq_set = General.freq_center + ...
                        General.bandwidth ./ General.num_sc .* (freq_index-1-(General.num_sc-1)./2);
    General.sc_interval = General.bandwidth ./ General.num_sc;
    
    Power.noise_power = Power.noise_density * General.sc_interval; % W
    Power.trans_power_per_sc = BS.trans_power ./ General.num_sc;
    Power.dynamicNoise_power = AIRS.dynamicNoise_power_density * General.sc_interval;
    
    %% channel
    
    [AIRS, PathLoss, Power] = gene_channel_para(BS, AIRS, PathLoss, Power, Direction, Distance);
    
    %% Element Increase
    
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
        
        tic
        fprintf("[Info] Array response vector calculating...\n");
        RotateMatrix = calRotateMatrix(AIRS.num, AIRS.horizon_rot_angle, zeros(AIRS.num,1), zeros(AIRS.num,1));
        [ARV_incident, ARV_reflect, ARV_y, ARV_z] = calARV(AIRS, Direction, General.freq_set, RotateMatrix);
        time = toc;
        fprintf("[Info] Array response vector calculating done! Cost time: %.4f s\n", time);
        
        %% Phase Optimization
        
        num_sc = General.num_sc;
        num_path_AIRStoUE = General.num_path_AIRStoUE;
        max_delay = General.max_delay;
        funcDelay = General.funcDelay;
        freq_set = General.freq_set;
        ERP_AIRStoUE = Power.ERP_AIRStoUE;

        tic
        ERP_AIRStoUE_const = parallel.pool.Constant(ERP_AIRStoUE);
        CCM = zeros(AIRS.M, AIRS.M, num_sc, num_angleTrial);
        parfor sc = 1:num_sc
        
            % fprintf("[Info] subcarrier %d is calculating...\n", sc);
            ARV_incident_sc = ARV_incident(:,sc);
            
            R_mat = zeros(num_path_AIRStoUE, num_path_AIRStoUE, num_angleTrial);
            for nat = 1:num_angleTrial
                normal_PL_fading_AIRStoUE = sqrt(1/2).*(randn(num_path_AIRStoUE,num_fading) + 1j*randn(num_path_AIRStoUE,num_fading));
                delay_AIRStoUE = funcDelay(freq_set(sc), rand(num_path_AIRStoUE,num_fading).*max_delay);
                h_AIRStoUE = normal_PL_fading_AIRStoUE .* delay_AIRStoUE;
                
                for numf = 1:num_fading
                    R_mat(:,:,nat) = R_mat(:,:,nat) + h_AIRStoUE(:,numf)*h_AIRStoUE(:,numf)';
                end
                R_mat(:,:,nat) = R_mat(:,:,nat) ./ num_fading; % Expectation
                
                sqrt_ERP_AIRStoUE = diag(sqrt(ERP_AIRStoUE_const.Value(:,nat))); % num_path x num_path
                ARV_reflect_sc_nat = ARV_reflect(:,:,sc,nat); % M x num_path
                CCM(:,:,sc,nat) = diag(ARV_incident_sc')*conj(ARV_reflect_sc_nat)*...
                                    sqrt_ERP_AIRStoUE' * R_mat(:,:,nat) * sqrt_ERP_AIRStoUE*...
                                  ARV_reflect_sc_nat.'*diag(ARV_incident_sc);
            end

        end
        MCCM = mean(CCM,3);
        time = toc;
        fprintf("[Info] calculate the CCM done! Cost time: %.4f s\n", time);

        % debugger
        % dc_sca version
        if debug == 1
            ARV_yz = zeros(AIRS.M, num_sc);
            CCM_test = zeros(AIRS.M, AIRS.M, num_sc);
            for sc = 1:num_sc
                ARV_yz(:,sc) = kron(ARV_y(:,sc) , ARV_z(:,sc) );
                CCM_test(:,:,sc) = conj(ARV_yz(:,sc)) * ARV_yz(:,sc).' * Power.ERP_AIRStoUE * Power.ERP_BStoAIRS;
                diff_ccm = CCM_test(:,:,sc) - CCM(:,:,sc); % e-20
            end
        end
        
        %----------------------- METHOD1: MCCM (对CCM做平均) -----------------------%
        % 从S_mccm看出只取第一列特征向量无法代表包含CCM所有特征
        phi_MCCM_SVD = zeros(AIRS.M,num_angleTrial);
        for nat = 1:num_angleTrial
            [U_mccm,S_mccm,~] = svd(MCCM(:,:,nat));
            phi_MCCM_SVD(:,nat) = exp(-1j*(angle(U_mccm(:,1)))); % phase extraction
        end
        
        %----------------------- METHOD2: Align center freq (单频点单位置)-----------------------%
        % 从S_mccm看出只取第一列特征向量无法代表包含CCM所有特征
        freq_center_idx = round(num_sc/2);
        phi_align_fc = zeros(AIRS.M, num_angleTrial);
        for nat = 1:num_angleTrial
            [U_ccm_fc,S_ccm_fc,~] = svd(CCM(:,:,freq_center_idx,nat));
            phi_align_fc(:,nat) = exp(-1j*(angle(U_ccm_fc(:,1)))); % phase extraction
        end

        %----------------------- METHOD3: Random Phase -----------------------%
        phi_rand = exp(1j*2*pi*rand(AIRS.M,num_angleTrial));

        % %----------------------- METHOD4: RCG algorithm -----------------------%
        % % probably problem: accuracy, normalization
        % path = ['\LoS_NLoS_phi_rcg_',num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z), ...
        %         '_BW_', num2str(General.bandwidth/1e6), 'MHz_fc_', num2str(General.freq_center/1e9), 'GHz.mat'];
        % file_path = [save_data_path, path];
        % if save_rcg_data == 1
        %     accuracy = 2e-6;
        %     phi_rcg = zeros(AIRS.M,num_angleTrial);
        %     for nat = 1:num_angleTrial
        %         fprintf("[Info] RCG algorithm is starting, angleTrial %d\n", nat)
        %         meet_accuracy = false;
        %         while ~meet_accuracy
        %             [phi_rcg(:,nat),meet_accuracy,two_norm_RG,objval_arr] = RCGalgorithm(CCM(:,:,:,nat),accuracy);
        %             fprintf("[Info] last two_norm_RG: %f\n", two_norm_RG*1e4);
        %         end
        %     end
        %     save(file_path,"phi_rcg");
        % else
        %     load(file_path);
        % end
        
        % %----------------------- METHOD5: DC-SCA algorithm -----------------------%
        % Phi_dc_sca = zeros(M,M,num_angleTrial);
        % for nat = 1:num_angleTrial
        %     Phi_init = exp(1j*2*pi*rand(M,1));
        %     Phi_dc_sca(:,:,nat) = NLoS_dc_sca_algorithm(CCM(:,:,:,nat), Phi_init, normal_radiaPattern_BStoAIRS*normal_radiaPattern_AIRStoUE, num_sc, normalization);
        % end

        %% theory formula

        sumSE_theory_approx = zeros(num_angleTrial, 1);
        for nat = 1:num_angleTrial
        
            % direct link
            mean_power_delay_d = 1;
            powGain_direct = PathLoss.PL_BStoUE * (1/mean_power_delay_d);
        
            % cascaded link
            const_term = AIRS.M^2 / num_path_AIRStoUE * AIRS.amplify_factor^2 * PathLoss.PL_BStoAIRS * PathLoss.PL_AIRStoUE * Power.ERP_BStoAIRS;
            channel_gain = zeros(num_sc,1);
            for sc = 1:num_sc
                channel_gain(sc) = abs(phi_rand(:,nat)' * CCM(:,:,sc,nat) * phi_rand(:,nat));
            end
            avg_term = mean(channel_gain);
            powGain_cascade =  const_term * avg_term;
        
            % ergodic rate
            signalPart = Power.trans_power_per_sc * (powGain_direct + powGain_cascade);
            noisePart = AIRS.M * PathLoss.PL_AIRStoUE * mean(Power.ERP_AIRStoUE(:,nat)) * AIRS.amplify_factor^2 * Power.dynamicNoise_power + Power.noise_power;
            sumSE_theory_approx(nat) = num_sc * log2(1 + signalPart / noisePart);
        end
        avg_sumSE_theory_approx(num_elem_idx,num_sc_idx) = mean(sumSE_theory_approx, 1);
        
        %% MC simulation

        tic

        num_method = config.num_method;
        num_path_BStoUE = General.num_path_BStoUE;
        num_path_AIRStoUE = General.num_path_AIRStoUE;
        light_speed = General.light_speed;

        M = AIRS.M;
        amplify_factor = AIRS.amplify_factor;

        trans_power_per_sc = Power.trans_power_per_sc;
        dynamicNoise_power = Power.dynamicNoise_power;
        noise_power = Power.noise_power;
        ERP_BStoAIRS = Power.ERP_BStoAIRS;

        PL_BStoAIRS = PathLoss.PL_BStoAIRS;
        PL_AIRStoUE = PathLoss.PL_AIRStoUE;

        dist_BStoAIRS_3D = Distance.dist_BStoAIRS_3D;
        
        ergodic_SE_MC = zeros(num_sc, num_method, num_angleTrial);
        sumSE_MC = zeros(num_method, num_angleTrial);
        
        for nat = 1:num_angleTrial
        
            ARV_reflect_sc_nat = ARV_reflect(:,:,:,nat);
            ERP_AIRStoUE_temp = ERP_AIRStoUE(:,nat);
        
            Phi_MCCM_SVD = diag(phi_MCCM_SVD(:,nat));
            Phi_align_fc = diag(phi_align_fc(:,nat));
            Phi_rand = diag(phi_rand(:,nat));
            % Phi_rcg = diag(phi_rcg(:,nat));
            % Phi_dc_sca = diag(phi_dc_sca(:,nat));

            SE_MC = zeros(num_sc, num_method, num_fading);
            parfor numf = 1:num_fading
        
                %------------------ amplitude + fading ------------------%
                PL_fading_BStoAIRS = sqrt(PL_BStoAIRS/2) .* (randn(num_path_BStoUE,1)+1j*randn(num_path_BStoUE,1)); 
                PL_fading_AIRStoUE = sqrt(PL_AIRStoUE/2) .* (randn(num_path_AIRStoUE,1)+1j*randn(num_path_AIRStoUE,1));
        
                %------------------ Delay ------------------%
                delay_BStoUE = rand(num_path_BStoUE, 1) .* max_delay;
                delay_BStoAIRS = dist_BStoAIRS_3D / light_speed;
                delay_AIRStoUE = rand(num_path_AIRStoUE, 1) .* max_delay;
        
                %------------------ Ergodic SE ------------------%
                SE_MC_numf = zeros(num_sc, num_method);
                for sc = 1:num_sc
                    h_BStoUE_sc = sum( sqrt(1/num_path_BStoUE) .* PL_fading_BStoAIRS .* funcDelay(freq_set(sc),delay_BStoUE) ); % 1 x 1
                    h_BStoAIRS_sc = sqrt(M) * sqrt(PL_BStoAIRS) * funcDelay(freq_set(sc),delay_BStoAIRS) * sqrt(ERP_BStoAIRS) * ARV_incident(:,sc); % M x 1
                    h_AIRStoUE_sc = sqrt(M/num_path_AIRStoUE) * ARV_reflect_sc_nat(:,:,sc) * (PL_fading_AIRStoUE .* funcDelay(freq_set(sc),delay_AIRStoUE) .* sqrt(ERP_AIRStoUE_temp)); % M x 1
                    
                    func_signalPow = @(Phi)trans_power_per_sc .* abs(h_BStoUE_sc + h_AIRStoUE_sc.' * amplify_factor * Phi * h_BStoAIRS_sc)^2;
                    func_noisePower = @(Phi)norm(h_AIRStoUE_sc.' * Phi)^2 * amplify_factor^2 * dynamicNoise_power + noise_power;
        
                    %----------------------- METHOD1: Phase optimization By JiangTao -----------------------%
                    SE_MC_numf(sc,1) = log2(1 + func_signalPow(Phi_MCCM_SVD) / func_noisePower(Phi_MCCM_SVD));
        
                    %----------------------- METHOD2: Phase Align MCCM corresponding to fc -----------------------%
                    SE_MC_numf(sc,2) = log2(1 + func_signalPow(Phi_align_fc) / func_noisePower(Phi_align_fc));
        
                    %----------------------- METHOD3: random phase -----------------------%
                    SE_MC_numf(sc,3) = log2(1 + func_signalPow(Phi_rand) / func_noisePower(Phi_rand));

                    % %----------------------- METHOD4: RCG algorithm -----------------------%
                    % R_MC_numf(sc,4) = log2(1 + func_signalPow(Phi_rcg) / func_noisePower(Phi_rcg));
        
                    % %----------------------- METHOD5: DC_SCA algorithm -----------------------%
                    % SE_MC_numf(sc,5) = log2(1 + func_signalPow(Phi_dc_sca) / func_noisePower(Phi_dc_sca));

                end
                SE_MC(:,:,numf) = SE_MC_numf;
            end
            ergodic_SE_MC(:,:,nat) = mean(SE_MC, 3); % num_sc x num_method
            sumSE_MC(:,nat) = sum(ergodic_SE_MC(:,:,nat), 1)';
        end
        time = toc;
        fprintf("[Info] MC simulation realization done! Cost time: %.4f s\n", time);

        %% validate formula

        avg_sumSE_MC(num_elem_idx,num_sc_idx) = mean(sumSE_MC(1,:), 2); % mean to angle
        avgDiff = avg_sumSE_MC(num_elem_idx,num_sc_idx) - avg_sumSE_theory_approx(num_elem_idx,num_sc_idx);
        fprintf("[Info] Diff between MC and theory formula is %.4f\n", avgDiff);

    end % end element increment

end % end sc increment

%% Save data

path = ['LoS_NLoS_AIRS_elem_from_',num2str(M_Y_sets(1)), 'x', num2str(M_Y_sets(1)), ...
        'to',num2str(M_Y_sets(end)), 'x', num2str(M_Y_sets(end)),...
        'SC_from_',num2str(num_sc_sets(1)),'to',num2str(num_sc_sets(end)),...
        '_BW_',num2str(General.bandwidth/1e6),'MHz_fc_',num2str(General.freq_center/1e9),'GHz_path_',num2str(num_path_AIRStoUE),'.mat'];
plot_avg_sumSE = [avg_sumSE_MC avg_sumSE_theory_approx];
save(fullfile(save_data_path, path),'plot_avg_sumSE');

%% Plot

freq = freq_set ./ 1e9;
finalRes = mean(ergodic_SE_MC, 3);

linewidth = config.linewidth;
color = config.color;
fontname = config.fontname;
fontsize = config.fontsize;

figure(1);
set(gcf,'color','w')
plot(freq', finalRes(:,1), 'color', color(1), 'linewidth', linewidth);
hold on
plot(freq', finalRes(:,2), 'color', color(2), 'linewidth', linewidth);
plot(freq', finalRes(:,4), 'color', color(3), 'linewidth', linewidth);
legend("MCCM-SVD","Align-fc","Random",'location','best');
if General.bandwidth/1e9 <= 1
    titlestr = [num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z),',BW: ', num2str(General.bandwidth/1e6),'MHz'];
else
    titlestr = [num2str(AIRS.M_Y),'x',num2str(AIRS.M_Z),',BW: ', num2str(General.bandwidth/1e9),'GHz'];
end
title(titlestr, 'FontName', fontname);
xlabel("Frequency(GHz)", 'FontName', fontname);
ylabel("Spectral Efficiency (bps/Hz)", 'FontName', fontname);
set(gca, 'fontsize', fontsize);

save(fullfile(save_data_path, 'LOSNLoSData.mat'), "finalRes");
