import os
import numpy as np

class Config():
    def __init__(self):
        cur_folder = os.path.dirname(__file__)
        self.data_dir = os.path.join(cur_folder, '..', 'result', 'data')
        self.figure_dir = os.path.join(cur_folder, '..', 'result', 'figure')
        self.model_dir = os.path.join(cur_folder, '..', 'result', 'model')

        self.num_vit_feat = 15  # vit
        self.num_vit_sample = 21
        self.num_classes = 3
        self.num_vit_output = self.num_vit_sample * self.num_classes

        self.num_cdf_case = 5  # cdf
        self.num_cdf_sample = 16
        self.num_cdf_output = 1

        self.batch_size = 64
        self.num_epoch = 500
        self.num_fold = 5

        self.lab = 6
        self.sublab = 30

        self.vit_start = False
        self.cdf_start = False
        self.MLP_start = False
        self.LSTM_start = False

        self.train_start = False
        self.test_start = False
        self.cal_start = False

        self.plot_cdf = False
        self.plot_loss = True

        self.numUE_list = np.linspace(20, 25, 2, dtype=int)
        self.model_name = 'vit.pth'
        self.loss_delta = 1.0

    def config(self, param_file):
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line[0] == '#':
                        continue
                    line = line.split(':')
                    if line[0] == 'num_vit_feat':
                        self.num_vit_feat = int(line[1])
                    elif line[0] == 'num_cdf_case':
                        self.num_cdf_case = int(line[1])
                    elif line[0] == 'num_vit_sample':
                        self.num_vit_sample = int(line[1])
                    elif line[0] == 'num_cdf_sample':
                        self.num_cdf_sample = int(line[1])
                    elif line[0] == 'num_vit_output':
                        self.num_vit_output = int(line[1])
                    elif line[0] == 'num_cdf_output':
                        self.num_cdf_output = int(line[1])
                    elif line[0] == 'batch_size':
                        self.batch_size = int(line[1])
                    elif line[0] == 'num_epoch':
                        self.num_epoch = int(line[1])
                    elif line[0] == 'num_fold':
                        self.num_fold = int(line[1])
                    elif line[0] == 'lab':
                        self.num_epoch = int(line[1])
                    elif line[0] == 'sublab':
                        self.num_fold = int(line[1])
                    elif line[0] == 'vit_start':
                        self.vit_start = bool(line[1])
                    elif line[0] == 'cdf_start':
                        self.cdf_start = bool(line[1])
                    elif line[0] == 'MLP_start':
                        self.MLP_start = bool(line[1])
                    elif line[0] == 'LSTM_start':
                        self.LSTM_start = bool(line[1])
                    elif line[0] == 'train_start':
                        self.train_start = bool(line[1])
                    elif line[0] == 'test_start':
                        self.test_start = bool(line[1])
                    elif line[0] == 'cal_start':
                        self.cal_start = bool(line[1])
                    elif line[0] == 'plot_cdf':
                        self.plot_cdf = bool(line[1])
                    elif line[0] == 'plot_loss':
                        self.plot_loss = bool(line[1])
