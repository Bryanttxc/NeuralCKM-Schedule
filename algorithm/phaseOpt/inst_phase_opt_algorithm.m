% created by Bryanttxc, FUNLab

function [SE_opt, time, phi_opt] = inst_phase_opt_algorithm(Power, BS_UE_link, BS_AIRS_link, AIRS_UE_link, ...
                                                        amplify_factor, phi_init, iter, numf)
%INSTANT_PHASE_OPT_ALGORITHM obtain the optimized AIRS phase
% cited by [1] 实现IRS相位优化算法
% [1] S. Zhang and R. Zhang, "Capacity Characterization for Intelligent Reflecting Surface Aided MIMO Communication," 
% in IEEE Journal on Selected Areas in Communications, vol. 38, no. 8, pp. 1823-1838, Aug. 2020, doi: 10.1109
%
% %%% input %%%
% Power: class
% BS_UE_link,BS_AIRS_link,AIRS_UE_link: channel response
% amplify_factor: AIRS amplify
% phi_init: initial phase coefficients
% %%% output %%%
% SE_opt: optimized ergodic SE
% time: calculation time
% phi_opt: optimized phase vector

%% Initialization

% first run need to setup, then can annotate
cvx_solver mosek
cvx_save_prefs

% channel
tmp_d = double(gather(squeeze(BS_UE_link))); % num_sc x 1
tmp_i = double(gather(squeeze(BS_AIRS_link))); % M x num_sc
tmp_r = double(gather(squeeze(AIRS_UE_link))); % M x num_sc
trans_power_per_sc = Power.trans_power_per_sc;
noisePart = sum(abs(tmp_r).^2, 1).' .* amplify_factor^2 .* Power.dynamicNoise_power + Power.noise_power;

% index for faster calculation
M = size(tmp_i, 1); % num of AIRS elements
arr_index = zeros(M, M-1);
for mm = 1:M
    arr_index(mm,:) = [1:mm-1 mm+1:M]';
end

% phase coefficients
num_sc = size(tmp_i, 2);
% phi_iter = repmat(exp(-1j*2*pi*rand(M,1)), 1, num_sc);
phi_iter = repmat(phi_init, 1, num_sc); % faster meet converge
phi_pre = phi_iter;

% other parameters
counter = 1;
cur_SE_opt = -200;
pre_SE_opt = 0;
SE_arr = []; % check for converge

%% Main Program

tic
while 1

    fprintf("[Info] Iteration %d, fading %d, round %d is calculating...\n", iter, numf, counter);

    for m = 1:M

        cvx_begin quiet

            % args
            variable phi_m complex
            expression phi(M,num_sc)
            expression cascade(num_sc,1)
            expression sumRate
            expression SE

            % phase coefficients
            index = [1:m-1 m+1:M]';
            phi(m,:) = phi_m;
            phi(index,:) = phi_iter(index,:);

            % direct part for faster
            h_d = tmp_d;
            h_ir_arr = amplify_factor .* tmp_r .* phi .* tmp_i; % cascaded channel
            h_ir = amplify_factor .* diag((tmp_r .* phi).' * tmp_i);
            direct = h_d .* conj(h_d) + h_d .* conj(h_ir) + h_ir .* conj(h_d); % num_sc x 1

            % cascade part for faster
            cascade_mulp_item = ( sum(tmp_r .* tmp_i, 1) .* conj(sum(tmp_r .* tmp_i, 1)) ).';

            for sc = 1:num_sc

                % % original version of direct part
                % h_d = BS_UE_link(:,:,sc); % direct channel
                % h_ir_arr = amplify_factor_inst .* AIRS_UE_link(:,:,sc).' .* phi .* BS_AIRS_link(:,:,sc); % cascaded channel
                % h_ir = sum(h_ir_arr); % sum for elements M
                % direct = h_d * h_d' + h_d * h_ir' + h_ir * h_d';

                % % original version of cascade part
                % cascade = cascade + AIRS_UE_link(:,:,sc) * BS_AIRS_link(:,:,sc) * BS_AIRS_link(:,:,sc)' * AIRS_UE_link(:,:,sc)';

                % cross part for faster
                h_ir_arr_tmp = h_ir_arr(:,sc);
                real_val = repmat(real(h_ir_arr_tmp),1,M-1) .* real( conj(h_ir_arr_tmp(arr_index)) ) - ( repmat(imag(h_ir_arr_tmp),1,M-1) .* imag( conj(h_ir_arr_tmp(arr_index)) ) );
                imag_val = repmat(imag(h_ir_arr_tmp),1,M-1) .* real( conj(h_ir_arr_tmp(arr_index)) ) + ( repmat(real(h_ir_arr_tmp),1,M-1) .* imag( conj(h_ir_arr_tmp(arr_index)) ) );
                cascade(sc) = sum(sum(real_val + 1j.*imag_val)); % cross item

                % % original version of cross part
                % cascade = 0;
                % for mm = 1:M
                %     index = [1:mm-1 mm+1:M]';
                %     real_val = real(h_ir_arr(mm)) .* real(h_ir_arr(index)') - ( imag(h_ir_arr(mm)) .* imag(h_ir_arr(index)') );
                %     imag_val = imag(h_ir_arr(mm)) .* real(h_ir_arr(index)') + ( real(h_ir_arr(mm)) .* imag(h_ir_arr(index)') );
                %     cascade = cascade + sum(real_val + 1j.*imag_val); % cross item
                % end
            end

            signalPart = trans_power_per_sc .* (direct + cascade + cascade_mulp_item);
            for sc = 1:num_sc
                sumRate = sumRate + ( -rel_entr(1, 1 + real(signalPart(sc)) / noisePart(sc)) / log(2) );
            end
            SE = sumRate / num_sc;

            % obj func
            maximize(SE)

            % constraints
            abs(phi_m) <= 1

        cvx_end

        % fprintf("Optimal Value(cvx_optval): %.4f\n", SE);
        if SE - cur_SE_opt > 0
            cur_SE_opt = SE;
            phi_iter = phi;
        elseif SE - cur_SE_opt == 0
            tolerate = tolerate + 1;
            if tolerate >= 5
                break;
            end
        end

    end % for m = 1:M

    % stop if meet condition
    if cur_SE_opt - pre_SE_opt < 0
        SE_opt = pre_SE_opt;
        phi_iter = phi_pre;
        break;
    elseif cur_SE_opt - pre_SE_opt <= 0.1
        SE_opt = cur_SE_opt;
        break;
    end

    % SE_arr = [SE_arr cur_SE];
    phi_pre = phi_iter;
    pre_SE_opt = cur_SE_opt;
    tolerate = 0;
    counter = counter + 1;

end % end while
time = toc;

phi_opt = phi_iter;

end
