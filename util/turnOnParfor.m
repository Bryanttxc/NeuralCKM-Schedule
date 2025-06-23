% created by Bryanttxc, FUNLab

function [pool,gpu] = turnOnParfor(config)
%TURNONPARFOR turn on parfor
%%% output
% pool: the parfor struct
% gpu: the gpu struct

if config.pick_parfor == 1
    maxNumWorkers = maxNumCompThreads;
    numWorkers = min(config.num_workers, maxNumWorkers);
    flag = gcp('nocreate');
    if isempty(flag)
        pool = parpool(numWorkers);
    else
        pool = gcp();
    end
    gpu = gpuDevice(1);
else
    fprintf("[Info] need set config.pick_pafor = 1 if want to use parfor\n");
end

end
