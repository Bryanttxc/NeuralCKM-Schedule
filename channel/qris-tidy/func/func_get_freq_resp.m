%{
  Authors:  Ilya Burtakov, Andrey Tyarin

  Wireless Networks Lab, Institute for Information Transmission Problems 
  of the Russian Academy of Sciences and 
  Telecommunication Systems Lab, HSE University.

  testbeds@wireless.iitp.ru

  You may get the latest version of the QRIS platform at
        https://wireless.iitp.ru/ris-quadriga/

  If you are using QRIS, please kindly cite the paper 

  I. Burtakov, A. Kureev, A. Tyarin and E. Khorov, 
  "QRIS: A QuaDRiGa-Based Simulation Platform for Reconfigurable Intelligent Surfaces," 
  in IEEE Access, vol. 11, pp. 90670-90682, 2023, doi: 10.1109/ACCESS.2023.3306954.     

  https://ieeexplore.ieee.org/document/10225307
%}

function [tmp_direct_signalPow, ...
          tmp_cascade_signalPow, ...
          tmp_scatter_signalPow, ...
          tmp_dynamic_noisePow, ...
          ergodic_SE_MC] = func_get_freq_resp(b, num_AIRS, AIRS_set, TxNAnt, RxNAnt, ...
                                              all_NSubcar, all_Bandwidth, sc_interval,...
                                              num_sc, num_fading, ...
                                              trans_power_per_sc, noise_power)
    %%% INPUT:
    %   b         - QuaDRiGa builder
    %   num_AIRS  - number of AIRSs
    %   AIRS_set  - set of AIRSs
    %   TxNAnt    - number of Tx antennas
    %   RxNAnt    - number of Rx antennas
    %   all_NSubcar - index of total subcarriers
    %   all_Bandwidth - whole bandwidth
    %   UE_NSubcar   - index of subcarriers allocated to UE
    %   UE_Bandwidth - bandwidth allocated to UE
    %   sc_interval - intervals of subcarrier
    %   num_sc - number of subcarriers
    %   num_fading - number of fading realizations
    %   trans_power_per_sc - transmit power per subcarrier
    %   noise_power - AWGN

    %% result array
    tmp_direct_signalPow = zeros(num_fading, 1);
    tmp_cascade_signalPow = zeros(num_AIRS, num_fading);
    tmp_scatter_signalPow = zeros(num_AIRS, num_fading);
    tmp_dynamic_noisePow = zeros(num_AIRS, num_fading);
    SE_MC_inst = zeros(num_AIRS+1, num_fading); % 1 for served by BS directly

    %% Main program
    % step1 generate large-scale parameters
    gen_parameters(b,0); % clear param. for ASD, ASA etc.
    gen_parameters(b,4); % gen all channel param.
    
    % AIRS_set = parallel.pool.Constant(AIRS_set);
    for numf = 1:num_fading
        
        cascade_item = zeros(num_sc, num_AIRS);
        scatter_item = zeros(num_sc, num_AIRS);
        
        % step2 generate small-scale parameters
        gen_parameters(b,2); % usage == 2
        
        % step3 generate channel coeff of each link
        c = b.get_channels;
        c_TxRx  = c(1); % TX to UE
        c_TxRIS = c(2:1+num_AIRS); % TX to all AIRSs
        c_RISRx = c(1+num_AIRS+1:end); % all AIRSs to UE
        
        % dimension: RxNAnt x TxNAnt x NSubcar
        H_TxRx_f = c_TxRx(1,1).fr(all_Bandwidth, all_NSubcar); % 1 x 1 x num_sc
        
        %% direct power
        
        tmp_d = squeeze(H_TxRx_f); % num_sc x 1
        tmp_direct_signalPow(numf,1) = mean(conj(tmp_d) .* tmp_d, 1);
        
        %% cascaded power
        
        dyn_noi_item = 0;
        for idx = 1:num_AIRS
            
            cur_IRS = AIRS_set{idx};
            
            % Question2: is it necessary to create two H_TxRIS?
            % H_TxRIS_wf = c_TxRIS(1, idx).fr(all_Bandwidth, all_NSubcar); % M x 1 x num_sc
            H_TxRIS_f = c_TxRIS(1, idx).fr(all_Bandwidth, all_NSubcar); % M x 1 x num_sc
            H_RISRx_f = c_RISRx(1, idx).fr(all_Bandwidth, all_NSubcar); % 1 x M x num_sc

            %%% Amplification factor (Attention: whole bandwidth!)
            sum_norm_T = trans_power_per_sc .* sum(abs(squeeze(H_TxRIS_f)).^2, 1); % 1 x num_sc
            amplify_factor_inst = sqrt( cur_IRS.amplify_power ./ ...
                                        (sum(sum_norm_T) + cur_IRS.M * cur_IRS.dynamic_noise_power_per_sc(sc_interval)) );
            
            %%% phase optimization
            % test for instant_MCCM
            % tmp_instant_CCM = zeros(cur_IRS.M, cur_IRS.M, num_sc);
            % for sc = 1:num_sc
            %     tmp_instant_CCM(:,:,sc) = H_RISRx_f(:,:,sc).' .* (H_TxRIS_f(:,:,sc) * H_TxRIS_f(:,:,sc)') .* conj(H_RISRx_f(:,:,sc));
            % end
            % tmp_instant_MCCM = mean(tmp_instant_CCM, 3); % same as instant_MCCM

            H_TxRIS_f = gpuArray(H_TxRIS_f);
            H_RISRx_f = gpuArray(H_RISRx_f);
            instant_CCM = permute(H_RISRx_f, [2,1,3]) .* (H_TxRIS_f .* permute(conj(H_TxRIS_f), [2,1,3])) .* conj(H_RISRx_f);
            instant_MCCM = mean(instant_CCM, 3); % mean operation for avg subcarriers
            instant_MCCM = double(gather(instant_MCCM));

            [U_inst_mccm,~,~] = svd(instant_MCCM);
            phi_beamform = exp(-1j*(angle(U_inst_mccm(:,1)))); % passive beamforming
            phi_scatter  = exp(-1j*(2*pi*rand(cur_IRS.M,1))); % random scattering
            
            % tmp_cascade_signalPow(idx,numf) = abs(amplify_factor_inst^2 * phi_beamform' * instant_MCCM * phi_beamform); % little claw expression
            tmp_cascade_signalPow(idx,numf) = abs(amplify_factor_inst^2 * phi_beamform.' * instant_MCCM * conj(phi_beamform)); % is_beamform = 0
            
            % tmp_scatter_signalPow(idx,numf) = abs(amplify_factor_inst^2 * phi_scatter' * instant_MCCM * phi_scatter); % little claw expression
            tmp_scatter_signalPow(idx,numf) = abs(amplify_factor_inst^2 * phi_scatter.' * instant_MCCM * conj(phi_scatter)); % is_beamform = 1
            
            %% dynamic noise

            % sum operation: M x num_sc -> 1 x num_sc
            % mean operation: 1 x num_sc -> 1 x 1
            tmp_dynamic_noisePow(idx,numf) = mean( sum(abs(squeeze(H_RISRx_f)).^2, 1) .* amplify_factor_inst^2 .* cur_IRS.dynamic_noise_power_per_sc(sc_interval) );

            %%% conserve variables, need check!
            tmp_i = squeeze(H_TxRIS_f); % M x num_sc
            tmp_r = squeeze(H_RISRx_f); % M x num_sc
            cascade_item(:,idx) = amplify_factor_inst .* diag((tmp_r .* phi_beamform).' * tmp_i); % num_sc x 1
            scatter_item(:,idx) = amplify_factor_inst .* diag((tmp_r .* phi_scatter).' * tmp_i); % num_sc x 1
            dyn_noi_item = dyn_noi_item + ...
                            sum(abs(squeeze(H_RISRx_f)).^2, 1)' .* amplify_factor_inst^2 .* cur_IRS.dynamic_noise_power_per_sc(sc_interval); % num_sc x 1
        end
        
        %% Ergodic sum Rate by Monte Carlo
        % served by AIRS
        noisePart_mat = dyn_noi_item + noise_power; % num_sc x 1
        for idx = 1:num_AIRS
            op = [1:idx-1 idx+1:num_AIRS];
            signalPart_mat = trans_power_per_sc .* abs( tmp_d + cascade_item(:,idx) + sum(scatter_item(:,op), 2) ).^2; % num_sc x 1
            SE_MC_inst(idx,numf) = mean( log2(1 + signalPart_mat ./ noisePart_mat), 1 );
        end
        % served by BS
        signalPart_mat = trans_power_per_sc .* abs( tmp_d + sum(scatter_item, 2) ).^2; % num_sc x 1
        SE_MC_inst(end, numf) = mean( log2(1 + signalPart_mat ./ noisePart_mat), 1 );
    
    end
    
    %%% indicators
    ergodic_SE_MC = mean(SE_MC_inst, 2); % (num_AIRS + 1) x 1, each IRS plays the role of server once a time.
    
    % clear instant_MCCM H_TxRIS_f H_RISRx_f instant_CCM
end
