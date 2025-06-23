% created by Bryanttxc, FUNLab

function sample = randShuffle(input, sampleLen)
% RANDSHUFFLE random shuffle and extend the input into the length of
% sampleLen
input = reshape(repmat(input, ceil(sampleLen/length(input)), 1), [], 1);
R = randperm(length(input));
R = R(1:sampleLen);
sample = input(R);

end