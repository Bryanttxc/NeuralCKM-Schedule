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

classdef class_Rx_linear < class_Rx
    % Rx-subclass for linear array
    
    properties
        Antenn %Antenna of qd_arrayant class for Rx
    end

    methods
        function obj = class_Rx_linear(Position, NumberOfAntennas, wavelength, antenn_type, CentralFrequency)
            
            superargs{1} = Position;
            
            obj = obj@class_Rx(superargs{:});
            
            % element postion (local coordinate)
            distance = wavelength / 2;
            pos_matrix = zeros(NumberOfAntennas, 3);
            x_shift = (NumberOfAntennas + 1) * distance / 2;
            for i = 1:NumberOfAntennas
                pos_matrix(i, :) = [i * distance - x_shift, 0, 0];
            end
            Rx_pos_matrix = zeros(size(pos_matrix, 1), 3);
            Rx_pos_matrix(1:1:end, :) = pos_matrix;
            
            % antenna properties
            Rx_antenn = qd_arrayant(antenn_type);
            Rx_antenn.center_frequency = CentralFrequency;
            Rx_antenn.no_elements = NumberOfAntennas;
            Rx_antenn.element_position = Rx_pos_matrix';
            for i = 1 : NumberOfAntennas
                Rx_antenn.rotate_pattern(0, 'x', i); %pol1
            end
            
            obj.Antenn = Rx_antenn;
        end
    end
end


