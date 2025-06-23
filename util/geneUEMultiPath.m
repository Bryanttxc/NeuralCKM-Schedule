% created by Bryanttxc, FUNLab

function [w_u_l] = geneUEMultiPath(w_u, numUE, Lr)
% GENEUEMULTIPATH generate multipath for UE
% input
% w_i: locations of UE u in U
% numUE: number of UEs
% Lr: number of multi-path

randCord = 10.*rand(Lr,3,numUE)-5;
w_u_l = zeros(Lr,3,numUE);
for temp = 1:numUE
    w_u_l(:,:,temp) = w_u(temp,:) + randCord(:,:,temp);
end

end