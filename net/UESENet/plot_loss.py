
# 读取result/data/others/vit_loss.json文件
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.config import Config

open_file = os.path.join(os.path.dirname(__file__), 'result', 'data', 'others', 'vit_loss.json')
with open(open_file, 'r') as f:
    vit_loss = json.load(f)
    
train_loss_set = vit_loss['train_loss']
val_loss_set = vit_loss['val_loss']

for _, (train_loss_curve, val_loss_curve) in enumerate(zip(train_loss_set, val_loss_set)):

    plt.rcParams.update({'font.size': 15})
    plt.rcParams['font.family'] = 'Times New Roman'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_loss_curve, 'r', label='train loss')
    ax.plot(val_loss_curve, 'b--', label='validation loss')
    ax.set_title(f'Training and Validation Loss of NN-1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()
    plt.close(fig)
    