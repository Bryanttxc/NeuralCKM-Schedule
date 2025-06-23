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

classdef class_RIS_singlepol < class_RIS
    
    % RIS-subclass for one polarization scenario

    properties
        Rx_Antenn %Antenna of qd_arrayant class for RIS as Rx
        Tx_Antenn %Antenna of qd_arrayant class for RIS as Tx
    end
    
    methods
        function obj = class_RIS_singlepol(Position, RISNAnt_y, RISNAnt_z, wavelength, CentralFrequency, ...
                                           antenna_gain, q, xrot, yrot, zrot)
            
            superargs{1} = Position;
            
            obj = obj@class_RIS(superargs{:});

            distance = wavelength / 2; %distance between RIS UC
            pos_matrix = zeros(RISNAnt_y * RISNAnt_z, 3); %vector of coordinates of RIS UC
            
            %Initially RIS locates on YZ-plane
            y_shift = (RISNAnt_y + 1) * distance / 2; %Shift due to RIS size
            z_shift = (RISNAnt_z + 1) * distance / 2;
            for j = 1:RISNAnt_z
                for i = 1:RISNAnt_y
                    pos_matrix( ((j - 1) * RISNAnt_y + i), :) = [0, i * distance - y_shift, j * distance - z_shift];
                end
            end
            RIS_pos_matrix = zeros(size(pos_matrix, 1), 3);
            RIS_pos_matrix(1:end, :) = pos_matrix;
            
            %%% cxt: Pattern 1: Create COS-UC uni-polarized RIS as Rx
            Ain = sqrt(antenna_gain); Bin = 0; Cin = q; Din = 0;
            COS_UC_pol = qd_arrayant('Parametric', Ain, Bin, Cin, Din);
            COS_UC_pol.no_elements = RISNAnt_y * RISNAnt_z;
            COS_UC_pol.element_position = RIS_pos_matrix';
            COS_UC_pol.center_frequency = CentralFrequency;
            
            % rotation
            COS_UC_pol.rotate_pattern([xrot,yrot,zrot], 'xyz', [], 1);
            % COS_UC_pol.visualize(1);
            % 
            % % validate the correction of cos_UC_pol
            % COS_UC_pol1 = qd_arrayant('Parametric', Ain, Bin, Cin, Din);
            % COS_UC_pol1.rotate_pattern(xrot, 'x', [], 1);
            % COS_UC_pol1.rotate_pattern(yrot, 'y', [], 1);
            % COS_UC_pol1.rotate_pattern(zrot, 'z', [], 1);
            % COS_UC_pol1.visualize(1);

            % %%% cxt: Pattern 2: Create parametric uni-polarized RIS as Rx
            % Q_UC_pol  = qd_arrayant('patch');
            % Q_UC_pol.no_elements = RISNAnt_y * RISNAnt_z;
            % Q_UC_pol.element_position = RIS_pos_matrix';
            % Q_UC_pol.center_frequency = CentralFrequency;
            % 
            % % normalization
            % Q_UC_pol1_Pow = Q_UC_pol.Fa .* Q_UC_pol.Fa;
            % Q_UC_pol.Fa = Q_UC_pol.Fa * sqrt( antenna_gain / max(Q_UC_pol1_Pow(:)) );
            % 
            % % rotation
            % Q_UC_pol.rotate_pattern(zrot, 'z', [], 1);
            % Q_UC_pol.visualize(1);

            RIS_Antenna = COS_UC_pol; % cxt: Q_UC_pol
            obj.Tx_Antenn = RIS_Antenna;
            obj.Rx_Antenn = RIS_Antenna;
        end
        
    end
end

