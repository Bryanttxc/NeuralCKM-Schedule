% created by Bryanttxc, FUNLab

function Rate = calErgodicRate(ARV_i, phi, ARV_r, funcSNR, coeff_nonserve)
% CALERGODICRATE calculate the ergodic rate

numRB = size(ARV_i,4);
numFreq = size(ARV_i,3);
Phi = diag(phi);

funcSingVal = @(ARV_r,Phi,ARV_i)sqrt( (ARV_r.'*Phi*ARV_i)' * (ARV_r.'*Phi*ARV_i) );
Rate = 0;
for v = 1:numRB
    for nf = 1:numFreq
        SingularVal = funcSingVal(ARV_r(:,:,nf,v), Phi, ARV_i(:,:,nf,v));
%         [~,SingularVal,~] = svd(ARV_r(:,:,nf,v).' * Phi * ARV_i(:,:,nf,v));
        Rate = Rate + log2(1 + funcSNR(SingularVal, coeff_nonserve(v,nf)));
    end
end

end