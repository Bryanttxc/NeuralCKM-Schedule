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

function [l] = func_get_QuaDRiGa_layout(s, Tx, Rx, IRS, scenario_name_LOS, scenario_name_NLOS)

    %%% INPUT:  
    %   s                  - qd_simulation_parameters structure if simulation
    %   Tx                 - Tx-class device
    %   Rx                 - Rx-class device
    %   RIS                - RIS_dualpol/unipol-class device
    %   scenario_name_LOS  - LOS propagation condition
    %   scenario_name_NLOS - NLOS propagation condition

    %%% OUTPUT:
    % l - QuaDRiGa layout

    l = qd_layout(s);
    numBS = length(Tx);
    numUE = length(Rx);
    numIRS = length(IRS);

    l.no_tx = (numIRS + numUE) * numBS + numUE * numIRS; %这里应该是TX包括基站和IRS, rx包括IRS和UE
    l.no_rx = (numIRS + numUE) * numBS + numUE * numIRS;
    
    endIdx_tx = 1;
    endIdx_rx = 1;

    for bs_idx = 1:numBS
        tmp_BS = Tx{bs_idx};

        %position of tx-1(transmitter) and it's antenna type
        l.tx_track(1, endIdx_tx) = qd_track('linear', 0, 0); %track consist of one point
        l.tx_track(1, endIdx_tx).initial_position = tmp_BS.Position.';
        l.tx_track(1, endIdx_tx).name = ['BS', num2str(bs_idx)];
        l.tx_track(1, endIdx_tx).scenario = scenario_name_NLOS;
        l.tx_array(1, endIdx_tx) = tmp_BS.Antenn.copy();
        endIdx_tx = endIdx_tx + 1;
    end

    for ue_idx = 1:numUE
        tmp_UE = Rx{ue_idx};

        %position of rx-1(receiver) and it's antenna type
        l.rx_track(1, endIdx_rx) = qd_track('linear', 0, 0);
        l.rx_track(1, endIdx_rx).initial_position = tmp_UE.Position.';
        l.rx_track(1, endIdx_rx).name = ['UE', num2str(ue_idx)];
        l.rx_track(1, endIdx_rx).scenario = scenario_name_NLOS; % cxt: scenario only cares Rx
        l.rx_array(1, endIdx_rx) = tmp_UE.Antenn.copy();
        endIdx_rx = endIdx_rx + 1;
    end

    for irs_idx = 1:numIRS
        tmp_RIS = IRS{irs_idx};
        
        %position of tx->2(RIS as transmitter)
        l.tx_track(1, endIdx_tx) = qd_track('linear', 0, 0);
        l.tx_track(1, endIdx_tx).initial_position = tmp_RIS.Position.';
        l.tx_track(1, endIdx_tx).name = ['IRS', num2str(irs_idx)]; % 'asTx'
        l.tx_track(1, endIdx_tx).scenario = scenario_name_LOS;
        l.tx_array(1, endIdx_tx) = tmp_RIS.Tx_Antenn.copy();

        %position of rx->2(RIS as receiver)
        l.rx_track(1, endIdx_rx) = qd_track('linear', 0, 0);
        l.rx_track(1, endIdx_rx).initial_position = tmp_RIS.Position.';
        l.rx_track(1, endIdx_rx).name = ['IRS', num2str(irs_idx)]; % 'asRx'
        l.rx_track(1, endIdx_rx).scenario = scenario_name_LOS; % cxt: scenario only cares RISasRx
        l.rx_array(1, endIdx_rx) = tmp_RIS.Rx_Antenn.copy();

        endIdx_tx = endIdx_tx + 1;
        endIdx_rx = endIdx_rx + 1;
    end

%     l.pairing = [1 2 1 ; 1 1 2]; % tx_track ; rx_track

    tx_RIS_pairing = [ones(1, numIRS) ; (numUE+1:numUE+numIRS)];
    RIS_rx_pairing = [repmat((numUE+1:numUE+numIRS), 1, numUE) ; reshape(repmat(1:numUE, numIRS, 1), [], 1)'];
    l.pairing = [ones(1,numUE) tx_RIS_pairing(1,:) RIS_rx_pairing(1,:); 
                    (1:numUE)  tx_RIS_pairing(2,:) RIS_rx_pairing(2,:)];
end

