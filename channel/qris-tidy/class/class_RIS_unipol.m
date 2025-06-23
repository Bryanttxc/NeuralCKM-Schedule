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

classdef class_RIS_unipol < class_RIS
    % RIS-subclass for one polarization scenario
    
    properties
        Rx_Antenn %Antenna of qd_arrayant class for RIS as Rx
        Tx_Antenn %Antenna of qd_arrayant class for RIS as Tx
    end
    
    methods
        function obj = class_RIS_unipol(Position, RISNAnt_x, RISNAnt_z, wavelength, CentralFrequency)
            
            superargs{1} = Position;
            
            obj = obj@class_RIS(superargs{:});

            distance = wavelength / 2; %distance between RIS UC
            pos_matrix = zeros(RISNAnt_x * RISNAnt_z, 3); %vector of coordinates of RIS UC
            
            %Initially RIS locates on XZ-plane
            x_shift = (RISNAnt_x + 1) * distance / 2; %Shift due to RIS size
            z_shift = (RISNAnt_z + 1) * distance / 2;
            for j = 1:RISNAnt_z
                for i = 1:RISNAnt_x
                    pos_matrix( ((j - 1) * RISNAnt_x + i), :) = [i * distance - x_shift, 0, j * distance - z_shift];
                end
            end

            RIS_pos_matrix = zeros(size(pos_matrix, 1), 3);
            RIS_pos_matrix(1:1:end, :) = pos_matrix;


            %%% Create all-omni two-polarized RISasRx for CST-case
            q = 0.285;
            Ain = sqrt(pi); Bin = 0; Cin = 4 * q; Din = 0;

            RIS_Rx_antenn = qd_arrayant('parametric', Ain, Bin, Cin, Din);
            RIS_Rx_antenn.no_elements = RISNAnt_x * RISNAnt_z;
            RIS_Rx_antenn.element_position = RIS_pos_matrix';
            RIS_Rx_antenn.center_frequency = CentralFrequency;

            obj.Rx_Antenn = RIS_Rx_antenn;



            RIS_Tx_antenn = qd_arrayant('parametric', Ain, Bin, Cin, Din);
            RIS_Tx_antenn.no_elements = RISNAnt_x * RISNAnt_z;
            RIS_Tx_antenn.element_position = RIS_pos_matrix';
            RIS_Tx_antenn.center_frequency = CentralFrequency;

            obj.Tx_Antenn = RIS_Tx_antenn;
        end
    end
end

