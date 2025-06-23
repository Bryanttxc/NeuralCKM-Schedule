# UESENet
Pytorch实现基于Transformer架构的部署参数与用户遍历频谱效率的智能映射方案

## 运行过程
### 1. 导入数据集
从../../simulation/link_level_new/LoSNLoS_quadriga_sample_newest.m采集得到
命名为train_data.xlsx

### 2.  开始训练
可在config类修改配置参数，选择运行的网络（vit/cdf/mlp/lstm）和任务（train）：
``` bash
python main.py vit train
```

### 3.  开始测试
可在config类修改配置参数，选择运行的网络（vit/cdf/mlp/lstm）和任务（test）：
``` bash
python main.py vit test
```

### 4.  开始应用
可在config类修改配置参数，选择运行的网络（vit/cdf）和任务（cal）
此步需要等matlab在cal文件夹生成cal_data_lab_x_y_numUE_z.xlsx后，方可正常运行：
``` bash
python main.py vit cal
```
运行下述命令，还需要matlab生成cal_SE_data_1_lab_x_y_numUE_z.mat后，方可正常运行：
``` bash
python main.py cdf cal
```
