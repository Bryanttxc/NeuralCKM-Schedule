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

classdef class_Tx_planar < class_Tx
    % Tx-subclass for planar array
    
    properties
        Antenn %Antenna of qd_arrayant class for Tx
    end

    methods
        function obj = class_Tx_planar(Position, TxNumberOfAntennas_y, TxNumberOfAntennas_z,...
                                 wavelength, AziWidth, ElevWidth, F2BRatio, DownTilt, antenn_type, CentralFrequency)
            
            superargs{1} = Position;
            
            obj = obj@class_Tx(superargs{:});


            distance = wavelength / 2;
            pos_matrix = zeros(TxNumberOfAntennas_y * TxNumberOfAntennas_z,3);
            y_shift = (TxNumberOfAntennas_y+1)*distance / 2;
            z_shift = (TxNumberOfAntennas_z+1)*distance / 2;
                    
            for j=1:TxNumberOfAntennas_z  
                for i=1:TxNumberOfAntennas_y 
                    pos_matrix(((j - 1)*TxNumberOfAntennas_y + i),:) = [0,i*distance-y_shift,j*distance-z_shift];
                end
            end
                    
            Tx_pos_matrix = zeros(2 * size(pos_matrix, 1), 3);
            Tx_pos_matrix(1:2:end,:) = pos_matrix;
            Tx_pos_matrix(2:2:end,:) = pos_matrix;
                    
            % downtilt = 0, because antenna array is rotated by rotate_pattern
            Tx_antenn = qd_arrayant(antenn_type, AziWidth, ElevWidth, F2BRatio, 0);
            Tx_antenn.center_frequency = CentralFrequency;
            Tx_antenn.no_elements = 2 * TxNumberOfAntennas_z * TxNumberOfAntennas_y;
            for i = 1 : TxNumberOfAntennas_z * TxNumberOfAntennas_y 
                Tx_antenn.rotate_pattern(45, 'x', 2*(i-1)+1);
                Tx_antenn.rotate_pattern(-45, 'x', 2*i);
                Tx_antenn.rotate_pattern(DownTilt, 'y', 2*(i-1)+1);
                Tx_antenn.rotate_pattern(DownTilt, 'y', 2*i);
            end
            Tx_antenn.element_position = Tx_pos_matrix';

            obj.Antenn = Tx_antenn;
        end
    end
end


