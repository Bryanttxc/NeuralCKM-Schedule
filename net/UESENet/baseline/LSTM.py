import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from scipy.io import savemat
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import time
import json

import utils.dataloader as util_dataset
from nets.benchNet import LSTM_NN
from utils.NN_1 import earlyStopping

###### LSTM model ######

class train_impl():
    def __init__(self, dataset, bins, num_case, num_cdf_sample,
        num_output, batch_size, num_epoch, model_path, loss_path,
        lr=1e-4, weight_decay=1e-5, patience=20, delta=1.0):
        super(train_impl, self).__init__()

        self.dataset = dataset
        self.bins = bins
        self.num_case = num_case
        self.num_cdf_sample = num_cdf_sample
        self.num_output = num_output
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model_path = model_path
        self.loss_path = loss_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.delta = delta
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def K_fold_cross_validation(self, num_fold):

        val_loss_min = float('inf')  # 当前最优模型的loss
        best_model = None  # 最优模型
        results = []
        train_loss_set = []
        val_loss_set = []
        
        kfold = KFold(n_splits=num_fold, shuffle=True, random_state=42)
        
        # step0 训练集和验证集预处理
        train_subset_list = []
        val_subset_list = []
        for idx in range(len(self.dataset)):
            tmp_set = self.dataset[idx]
            for fold, (train_idx, val_idx) in enumerate(kfold.split(tmp_set)):
                train_subset_list.append(Subset(tmp_set, train_idx))
                val_subset_list.append(Subset(tmp_set, val_idx))

        for fold in range(num_fold):
            
            print(f"Fold {fold + 1} / {num_fold}")
            
            # step1 划分训练集和验证集
            train_loader_list = []
            val_loader_list = []
            for idx in range(fold, len(train_subset_list), num_fold):

                train_subset = train_subset_list[idx]
                val_subset = val_subset_list[idx]
                train_loader_list.append(DataLoader(train_subset, batch_size=self.batch_size, shuffle=True))
                val_loader_list.append(DataLoader(val_subset, batch_size=self.batch_size, shuffle=False))

            # step2 重新初始化模型，每一折从头开始训练
            model = LSTM_NN(bins=self.bins, device=self.device, cdf_dim=self.num_cdf_sample)
            if torch.cuda.is_available():
                cudnn.benchmark = True
                model = model.to(self.device)

            # step3 创建损失函数
            loss_fn = nn.HuberLoss(delta=self.delta)

            # step4 创建AdaW优化器
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # step5 创建调度器
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)

            # step6 创建早停策略
            early_stop = earlyStopping(patience=self.patience)

            # step7 训练模型
            train_loss_curve, val_loss_curve, best_model_per_fold, min_val_loss = self.train(model, train_loader_list, val_loader_list, loss_fn,
                                                                                        optimizer, lr_scheduler, early_stop)
            # step8 获取当前折的最低loss
            train_loss_set.append(train_loss_curve)
            val_loss_set.append(val_loss_curve)
            results.append(min_val_loss)
            print("Fold {:d} / {:d}, Loss: {:.4f}".format(fold+1, num_fold, min_val_loss))

            # step9 保存最低loss的模型
            if min_val_loss < val_loss_min:
                val_loss_min = min_val_loss
                best_model = best_model_per_fold

        # step10 存模型和loss曲线
        print("Best Loss: {:.4f}".format(val_loss_min))
        torch.save(best_model, self.model_path)
        with open(self.loss_path, 'w') as f:
            json.dump({'train_loss': train_loss_set, 'val_loss': val_loss_set, 'results': results}, f)

        # step11 绘出所有折的loss曲线
        for _, (train_loss_curve, val_loss_curve) in enumerate(zip(train_loss_set, val_loss_set)):
            self.plot_loss(train_loss_curve, val_loss_curve)

    # 学习率预热策略
    def lr_lambda(self, epoch):

        first_stage = 15.0
        second_stage = 100.0
        top_val = 8e-4
        diff_stage = second_stage - first_stage

        if epoch + 1 <= (int)(first_stage):
            return (9e-4 / 1e-4) * ((epoch + 1) / first_stage)
        elif (int)(first_stage) < epoch + 1 <= (int)(second_stage):
            decay_rate = top_val/1e-4
            return decay_rate **( ((int)(second_stage)-epoch) / diff_stage)
        else:
            return 1.0

    def train(self, model, train_loader_list, val_loader_list, loss_fn,
            optimizer, lr_scheduler, early_stop):

        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(self.num_epoch):
            print('Epoch: ', epoch)
            train_loss, val_loss = self.fit_one_epoch(model, train_loader_list, val_loader_list,
                                                    loss_fn, optimizer, self.device)
            lr_scheduler.step()

            train_loss_vec.append(train_loss)
            val_loss_vec.append(val_loss)

            if early_stop(val_loss, model):
                break

        best_model = early_stop.best_model
        min_val_loss = early_stop.min_loss

        return train_loss_vec, val_loss_vec, best_model, min_val_loss

    def fit_one_epoch(self, model, train_loader_list, val_loader_list,
                      loss_fn, optimizer, device):
        
        start_time = time.time()
        
        model.train()
        train_loss_list = []
        for train_loader in train_loader_list:
            train_data_size = len(train_loader)
            for step, data in enumerate(train_loader):
                feature, labels = data[0], data[1]  # 取出图片及标签
                feat_dim = feature.shape[1] // self.num_cdf_sample
                feature = feature.reshape([-1, feat_dim, self.num_cdf_sample])

                feature, labels = feature.to(device), labels.to(device)

                optimizer.zero_grad()  # 清零梯度
                outputs = model(feature)  # 前向传播
                loss_value = loss_fn(outputs, labels)  # 计算loss值
                loss_value.backward()  # 反向传播
                optimizer.step()  # 优化器迭代

                train_loss_list.append(loss_value.item())

                # 画进度条
                self.draw_process_bar(step, train_data_size, loss_value, 'train')

        print()
        print('train_loss:{:.3f}'.format(np.mean(train_loss_list)))

        model.eval()
        val_loss_list = []
        for val_loader in val_loader_list:
            val_data_size = len(val_loader)
            for step, data in enumerate(val_loader):
                feature, labels = data[0], data[1]
                feat_dim = feature.shape[1] // self.num_cdf_sample
                feature = feature.reshape([-1, feat_dim, self.num_cdf_sample])

                with torch.no_grad():
                    feature, labels = feature.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(feature)
                    loss_value = loss_fn(outputs, labels)

                    val_loss_list.append(loss_value.item())

            self.draw_process_bar(step, val_data_size, loss_value, 'validate')

        print()
        print('validate_loss:{:.3f}, epoch_time:{:.3f} s'.format(np.mean(val_loss_list), time.time()-start_time))

        return np.mean(train_loss_list), np.mean(val_loss_list)

    def draw_process_bar(self, step, length, loss_value, phase):

        rate = (step + 1) / length
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r{:s} loss: {:^3.0f}%[{}->{}]{:.3f}".format(
            phase, int(rate * 100), a, b, loss_value), end="")

    def plot_loss(self, train_loss_vec, test_loss_vec):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss_vec, 'r', label='train loss')
        ax.plot(test_loss_vec, 'b--', label='validation loss')
        ax.set_title(f'Training and Validation Loss of simpLSTM')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()
        plt.close(fig)


class train():
    def __init__(self, train_dataset, bins, num_case, num_sample,
            num_output, batch_size, num_epoch, model_path, loss_path):

        self.train_impl = train_impl(train_dataset, bins, num_case, num_sample, 
                    num_output, batch_size, num_epoch, model_path, loss_path)

    def K_fold_cross_validation(self, num_fold):
        
        self.train_impl.K_fold_cross_validation(num_fold=num_fold)


class test_impl():
    def __init__(self, bins, test_data_path, num_case, 
                 num_sample, num_output, batch_size, load_path,
                 delta=1.0):
        super(test_impl, self).__init__()

        test_loader_list = []
        for data_path in test_data_path:
            test_data = util_dataset.test_cdf_dataset(data_path, num_output, 
                                    cdf_dim=num_sample, var_name='test_result')
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            test_loader_list.append(test_loader)
        self.test_loader_list = test_loader_list

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_model = LSTM_NN(bins, device=self.device, cdf_dim=num_sample)

        self.criterion = nn.HuberLoss(delta=delta)
        self.num_case = num_case
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.load_path = load_path

    def measure_time(func):
        """装饰器: measure execute time of func"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            return result, end_time - start_time
        return wrapper
    
    @measure_time
    def process_data(self, test_model, data):
        """deal a batch of data"""
        feature, labels = data[0], data[1]
        feat_dim = feature.shape[1] // self.num_sample
        feature = feature.reshape([-1, feat_dim, self.num_sample])
        
        with torch.no_grad():
            feature, labels = feature.to(self.device), labels.to(self.device)
            outputs = test_model(feature)
            loss = self.criterion(outputs, labels)
            mae = torch.mean((outputs - labels).abs()).item()
            mre = torch.mean((outputs - labels).abs() / labels.abs()).item()
            return loss.item(), mae, mre

    def test(self, plot_loss=False):

        test_model = self.test_model
        test_model.to(self.device)
        test_model.load_state_dict(torch.load(self.load_path, weights_only=False))

        test_model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_mre = 0.0
        total_time = 0.0
        count = 0
        skipped_batches = 0
        
        outputs_set = []
        labels_set = []
        diff_set = []
        for test_loader in self.test_loader_list:
            for step, data in enumerate(test_loader):
                
                if len(data) != 2:
                    print("[Warning] Invalid data format, skipping this batch.")
                    skipped_batches += 1
                    continue
                
                # test data and measure time
                (loss, mae, mre), elapsed_time = self.process_data(test_model, data)
            
                if loss == float('inf') or mae == float('inf') or mre == float('inf'):
                    print(f"[Warning] Error occurred in batch {step}, skipping this batch.")
                    skipped_batches += 1
                    continue
                
                total_loss += loss
                total_mae += mae
                total_mre += mre
                total_time += elapsed_time
                count += 1
                
                # visualize curves of cdf
                if plot_loss:
                    feature, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = test_model(feature)
                    outputs_set.append(outputs)
                    labels_set.append(labels)
                    diff_set.append(outputs - labels)

        # prevent divide zero
        if count == 0:
            print("No valid data batches found.")
            return
        
        mean_time = total_time / count
        mean_loss = total_loss / count
        mean_mae = total_mae / count
        mean_mre = total_mre / count

        print('Avg calculate time: {:.4f}'.format(mean_time))
        print('Avg loss: {:.4f}'.format(mean_loss))
        print('MAE: {:.4f}'.format(mean_mae))
        print('MRE: {:.4f}'.format(mean_mre))
        
        if plot_loss:
            self.plot_loss_curve(outputs_set, labels_set, diff_set)

    def plot_loss_curve(self, outputs_set, labels_set, diff_set):

        outputs_set = torch.cat(outputs_set, dim=0)
        labels_set = torch.cat(labels_set, dim=0)
        diff_set = torch.cat(diff_set, dim=0)
        
        outputs_set = outputs_set.cpu().detach().numpy()
        outputs_set = outputs_set.squeeze(1)
        labels_set = labels_set.cpu().detach().numpy()
        labels_set = labels_set.squeeze(1)
        diff_set = diff_set.cpu().detach().numpy()
        diff_set = diff_set.squeeze(1)
        
        step = 5
        outputs_set_plot = outputs_set[::step]
        labels_set_plot = labels_set[::step]
        diff_set_plot = diff_set[::step]
        
        x = np.arange(0, len(outputs_set_plot))
        
        plt.rcParams.update({'font.size': 30})
        plt.rcParams['font.family'] = 'Times New Roman'
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.scatter(x, outputs_set_plot, linewidth=1, color='#0072B2', linestyle='--', label='predict')
        ax.scatter(x, labels_set_plot, linewidth=1, color='#D55E00', linestyle='--', label='truth')
        ax.vlines(x, ymin=0, ymax=outputs_set_plot, linewidth=1, color='#0072B2', linestyle='dashed', alpha=0.5)
        ax.vlines(x, ymin=0, ymax=labels_set_plot, linewidth=1, color='#D55E00', linestyle='dashed', alpha=0.5)
        # plt.title('1BS-2IRS-1UE scene, change UE position, BW=20MHz,SC=60KHz')
        plt.xlabel('Test Sample index')
        plt.ylabel('Ergodic Throughput (bps/Hz)')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.scatter(x, diff_set_plot, linewidth=1, color='blue', linestyle='--')
        ax.axhline(y=-0.5, color='gray', linestyle='--', label='y=0.5')
        ax.axhline(y=0.5, color='gray', linestyle='--', label='y=0.5')
        plt.xlabel('Test Sample index')
        plt.ylabel('Diff between Predict and Truth (bps/Hz)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()
        plt.close(fig1)
        plt.close(fig2)


class test():
    def __init__(self, bins, test_data_path, num_case, 
                 num_sample, num_output, batch_size, load_path):

        self.test_impl = test_impl(bins, test_data_path, num_case, 
                        num_sample, num_output, batch_size, load_path)

    def test(self, plot_loss=False):

        self.test_impl.test(plot_loss)
