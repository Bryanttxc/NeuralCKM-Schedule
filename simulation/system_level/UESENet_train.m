% created by Bryanttxc, FUNLab

function UESENet_train(NN_train)
%UESENET_TRAIN start UESENet train

root_path = [fileparts(fileparts(pwd)), '\'];
py_interpreter = 'C:\Users\Sherlock\.conda\envs\UESENet\python.exe'; % depend on your device
py_uesenet_main_path = [root_path, '\net\UESENet\main.py']; % depend on your device

tic
if NN_train == 1
    [~, NN_1_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
        ' --model vit --model vit.pth --task train --num_fold 9'])
elseif NN_train == 2
    [~, NN_2_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
        ' --model cdf --model cdf.pth --task train --num_fold 9'])
elseif NN_train == 3
    [~, NN_3_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
        ' --model mlp --model mlp.pth --task train --num_fold 9'])
elseif NN_train == 4
    [~, NN_4_res] = system([py_interpreter, ' ', py_uesenet_main_path, ...
        ' --model lstm --model lstm.pth --task train --num_fold 9'])
end
time = toc;
fprintf("[Info] UESENet train done! Cost time: %.4f", time);

end
