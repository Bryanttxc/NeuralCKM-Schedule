import json
import matplotlib.pyplot as plt
import numpy as np

# 读取json文件
def read_json(json_file):
    
    file = 'net/UESENet/result/data/others/' + json_file
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def plot_loss(train_loss_vec, test_loss_vec, name):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_loss_vec, 'r', label='training loss')
    ax.plot(test_loss_vec, 'b--', label='validation loss')
    ax.set_title(f'Training and Validation Loss of {name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    # 读取json文件
    data = read_json('cdf_loss.json')
    train_loss_set = data['train_loss']
    val_loss_set = data['val_loss']
    
    for _, (train_loss_curve, val_loss_curve) in enumerate(zip(train_loss_set, val_loss_set)):
            plot_loss(train_loss_curve, val_loss_curve, 'SE-Net')

