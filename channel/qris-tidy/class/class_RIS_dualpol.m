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

classdef class_RIS_dualpol < class_RIS
    
    % RIS-subclass for two polarization scenario

    properties
        Rx_Antenn %Antenna of qd_arrayant class for RIS as Rx
        Tx_Antenn %Antenna of qd_arrayant class for RIS as Tx
    end
    
    methods
        function obj = class_RIS_dualpol(Position, RISNAnt_y, RISNAnt_z, wavelength, CentralFrequency)
            
            superargs{1} = Position;
            
            obj = obj@class_RIS(superargs{:});

            distance = wavelength / 2; %distance between RIS UnitCell
            pos_matrix = zeros(RISNAnt_y * RISNAnt_z, 3); %vector of coordinates of RIS UC
            
            %Initially RIS locates on XZ-plane
            y_shift = (RISNAnt_y + 1) * distance / 2; %Shift due to RIS size
            z_shift = (RISNAnt_z + 1) * distance / 2;
            for j = 1:RISNAnt_z
                for i = 1:RISNAnt_y
                    pos_matrix( ((j - 1) * RISNAnt_y + i), :) = [0, i * distance - y_shift, j * distance - z_shift];
                end
            end

            RIS_pos_matrix = zeros(2 * size(pos_matrix, 1), 3);
            RIS_pos_matrix(1:2:end, :) = pos_matrix; % fds: 每个位置重复一次（相邻，双极化天线）
            RIS_pos_matrix(2:2:end, :) = pos_matrix;

            %%% Create all-omni two-polarized RIS as Rx for CST-case
            RIS_Rx_antenn = qd_arrayant;
            RIS_Rx_antenn.no_elements = 2 * RISNAnt_y * RISNAnt_z;
            RIS_Rx_antenn.element_position = RIS_pos_matrix';
            RIS_Rx_antenn.center_frequency = CentralFrequency;

            obj.Rx_Antenn = RIS_Rx_antenn;

            RIS_Tx_antenn = qd_arrayant; % fds: IRS有两个角色，作为接受方和发送方，相当于复制
            RIS_Tx_antenn.no_elements = 2 * RISNAnt_y * RISNAnt_z;
            RIS_Tx_antenn.element_position = RIS_pos_matrix';
            RIS_Tx_antenn.center_frequency = CentralFrequency;

            obj.Tx_Antenn = RIS_Tx_antenn;
        end
        
    end
end

