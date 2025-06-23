% created by Bryanttxc, FUNLab

function [multipath_theta, multipath_phi] = gene_multipath_angle(horizon_rotate_angle, multipath_AIRStoUE, numAngleTrial, index, seed)
%GENEMULTIPATH generate multipath angle
rand('state', seed);
multipath_theta = zeros(multipath_AIRStoUE, numAngleTrial);
multipath_phi = zeros(multipath_AIRStoUE, numAngleTrial);

if strcmp(index,'uniform')
    %% Uniform distribution generation
    for angtrial = 1:numAngleTrial
        multipath_theta(:,angtrial) = rand(multipath_AIRStoUE,1).*90 + 90; % vertical angle, 90~180
        multipath_phi(:,angtrial) = rand(multipath_AIRStoUE,1).*180 - 90; % horizontal angle
    end

elseif strcmp(index,'specific')
    %% Specific distribution generation
    % angular spread from scattered waves
    % cited by《A 3D geometry-based stochastic channel model for UAV-MIMO channels》
    % syms theta phi
    k               = 3;   % spreading control parameter
    theta_g         = -45; % mean angle -> vertical
    theta_m         = (pi/10)*180/pi; % variance angle -> vertical
    lower_cos       = theta_g - theta_m;
    upper_cos       = theta_g + theta_m;
    lower_von       = horizon_rotate_angle - 90;
    upper_von       = horizon_rotate_angle + 90;
    phi_mu          = horizon_rotate_angle; % mean angle -> horizontal

    % handle function
    funcCosinePdf = @(theta,theta_g,theta_m)pi/(4*theta_m) * cos(pi/2*(theta-theta_g)/theta_m);
    funcVonMisesPdf = @(phi,phi_mu,k)exp(k*cos(phi-phi_mu))/(2*pi*besseli(0,k));

    maxCosinePdf = pi./(4.*theta_m);
    maxVonMisesPdf = exp(k)/(2*pi*besseli(0,k));

    theta_MC = zeros(multipath_AIRStoUE,1);
    phi_MC = zeros(multipath_AIRStoUE,1);
    cosinePDF_MC = zeros(multipath_AIRStoUE,1);
    vonMisesPDF_MC = zeros(multipath_AIRStoUE,1);
    for angtrial = 1:numAngleTrial
        cnt1 = 0;cnt2 = 0;

        % Acceptance-Rejection Method
        % Step 1: generate a random number X uniformly distributed in the
        %         target domain
        % Step 2: calculate its corresponding target PDF value f(X)
        % Step 3: generate a random number U uniformly distributed in [0,m]
        %         where m = max(f(x))
        % Step 4: if U <= f(X), X will be chosen as the candidate number
        % Repeat Step 1-4 until satisfying the exit condition
        while cnt1 < multipath_AIRStoUE
            temp_theta = (upper_cos-lower_cos)*rand(1) + lower_cos;
            temp_cosinePDF = funcCosinePdf(temp_theta, theta_g, theta_m); % vertical distribution   
            randTheta = maxCosinePdf * rand(1);
            if randTheta <= temp_cosinePDF
                cnt1 = cnt1 + 1;
                theta_MC(cnt1) = 90 - temp_theta;
                cosinePDF_MC(cnt1) = temp_cosinePDF;
            end
        end

        while cnt2 < multipath_AIRStoUE
            temp_phi = (upper_von-lower_von)*rand(1) + lower_von;
            temp_vonMisesPDF = funcVonMisesPdf(temp_phi, phi_mu, k); % horizontal distribution
            randPhi = maxVonMisesPdf * rand(1);
            if randPhi <= temp_vonMisesPDF
                cnt2 = cnt2 + 1;
                phi_MC(cnt2) = temp_phi;
                vonMisesPDF_MC(cnt2) = temp_vonMisesPDF;
            end
        end

        multipath_dir(:,:,angtrial) = [sind(theta_MC).*cosd(phi_MC) sind(theta_MC).*sind(phi_MC) cosd(theta_MC)];
    end
else
    error("Have no this choice! please check the index.");
end

end

