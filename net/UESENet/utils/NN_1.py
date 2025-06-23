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
from nets.net import vit_NN

###### vit model ######

# Loss Function
class composite_loss(nn.Module):
    def __init__(self, device, num_cdf_sample = 21, delta = 1.0, 
            lambda_mask = 20.0, lambda_monotonic = 2.0, lambda_ks = 0.5):
        super().__init__()

        self.device = device
        self.num_cdf_sample = num_cdf_sample
        self.delta = delta
        self.lambda_mask = lambda_mask
        self.lambda_monotonic = lambda_monotonic
        self.lambda_ks = lambda_ks
        
        self.weights = [1.0, 1.0, 1.0]
        self.invalid_value = -600.0
        
        self.dir_indices = torch.tensor(range(0*num_cdf_sample, 1*num_cdf_sample)).to(device) # direct link
        self.cas_indices = torch.tensor(range(1*num_cdf_sample, 2*num_cdf_sample)).to(device) # cascade link
        self.dyn_indices = torch.tensor(range(2*num_cdf_sample, 3*num_cdf_sample)).to(device) # dynamic noise
        
        self.bce = nn.BCELoss()
        
        self.log_sigma_reg = nn.Parameter(torch.zeros(1)).to(device)
        self.log_sigma_mask = nn.Parameter(torch.zeros(1)).to(device)

    def AutoWeightedLoss(self, loss_regression, loss_mask):
        # regression loss 部分
        loss1 = (1.0 / (2.0 * torch.exp(self.log_sigma_reg)**2)) * loss_regression
        loss1 += self.log_sigma_reg  # 正则项

        # mask loss 部分
        loss2 = (1.0 / (2.0 * torch.exp(self.log_sigma_mask)**2)) * loss_mask
        loss2 += self.log_sigma_mask

        total_loss = loss1 + loss2
        return total_loss.squeeze()

    def huber_loss(self, y_pred, y_true):
        
        errors = y_true - y_pred
        abs_errors = torch.abs(errors)
        
        huber_loss = torch.where(
                abs_errors < self.delta,
                0.5 * errors**2,
                self.delta * (abs_errors - 0.5 * self.delta)
        )

        valid_mask = (y_true != self.invalid_value).float()
        huber_loss = huber_loss * valid_mask
        return huber_loss.sum() / (valid_mask.sum() + 1e-6)

    def monotonic_loss(self, y_pred, y_true):
        # 计算y_i - y_{i+1}
        diffs = y_pred[:, :-1] - y_pred[:, 1:]
        diffs_true = y_true[:, :-1] - y_true[:, 1:]
        mask = (y_true != self.invalid_value).float()
        valid_pairs = mask[:, :-1] * mask[:, 1:]
        
        slope_diff = torch.abs(diffs - diffs_true) * valid_pairs

        # 返回斜率误差的平均值（即 MAE）
        slope_loss = slope_diff.sum() / valid_pairs.sum().clamp(min=1.0)
    
        return slope_loss
    
        # order_loss = torch.relu(diffs) * valid_pairs
        # return order_loss.sum() / (valid_pairs.sum() + 1e-6)
        # order_loss = torch.relu(diffs).mean()  # punish negative diffs
        # return order_loss
        
    def ks_penalty(self, y_pred, y_true):
        mask = (y_true != self.invalid_value).float()
        diff = torch.abs(y_pred - y_true) * mask
        return torch.max(diff, dim=1).values.mean()

    def get_segment(self, y, indices):
        return torch.index_select(y, 1, indices)

    def forward(self, y_pred, y_true, pred_mask, true_mask):
        
        yp_dir = self.get_segment(y_pred, self.dir_indices)
        yt_dir = self.get_segment(y_true, self.dir_indices)
        
        yp_cas = self.get_segment(y_pred, self.cas_indices)
        yt_cas = self.get_segment(y_true, self.cas_indices)
        
        yp_dyn = self.get_segment(y_pred, self.dyn_indices)
        yt_dyn = self.get_segment(y_true, self.dyn_indices)

        # Huber loss
        loss_dir = self.huber_loss(yp_dir, yt_dir) + self.lambda_monotonic * self.monotonic_loss(yp_dir, yt_dir)
        loss_cas = self.huber_loss(yp_cas, yt_cas) + self.lambda_monotonic * self.monotonic_loss(yp_cas, yt_cas)
        loss_dyn = self.huber_loss(yp_dyn, yt_dyn) + self.lambda_monotonic * self.monotonic_loss(yp_dyn, yt_dyn)

        regression_loss = (  
            self.weights[0] * loss_dir +
            self.weights[1] * loss_cas +
            self.weights[2] * loss_dyn
        ) / sum(self.weights)
        
        # mask loss
        mask_loss = 0.0
        if true_mask is not None and pred_mask is not None:
            mask_loss = self.bce(pred_mask, true_mask)
            
        # total_loss = regression_loss + self.lambda_mask * mask_loss
        total_loss = self.AutoWeightedLoss(regression_loss, mask_loss)
        return total_loss


# CDF补齐
class CDFPatcher():
    def __init__(self, max_invalid_len=6, fill_method='linear', fallback_value=-600.0):
        assert fill_method in ['copy', 'linear', 'constant']
        self.max_invalid_len = max_invalid_len
        self.fill_method = fill_method
        self.fallback_value = fallback_value
        
    def needs_patch(self, row, threshold=-550.0, check_len=5):
        front = row[:check_len]
        return (front.mean() < threshold) or (front.std() < 1e-3)

    def patch(self, y_pred):
        B, N = y_pred.shape
        y_patched = y_pred.clone()

        for i in range(B):
            idx = self.max_invalid_len
            if self.fill_method == 'copy':
                y_patched[i, :idx] = y_patched[i, idx]
            elif self.fill_method == 'linear':
                if idx + 1 < N:
                    slope = y_patched[i, idx + 1] - y_patched[i, idx]
                else:
                    slope = 0.0
                for j in range(idx - 1, -1, -1):
                    y_patched[i, j] = y_patched[i, j + 1] - slope
            elif self.fill_method == 'constant':
                y_patched[i, :idx] = self.fallback_value
        return y_patched


# CDF评价指标
class CdfEvaluator():
    def __init__(self, device, num_cdf_sample=21, mask_val=-600.0):
        self.mask_val = mask_val
        self.num_sample = 21
        self.num_classes = 3
        
        self.dir_indices = torch.tensor(range(0*num_cdf_sample, 1*num_cdf_sample)).to(device) # direct link
        self.cas_indices = torch.tensor(range(1*num_cdf_sample, 2*num_cdf_sample)).to(device) # cascade link
        self.dyn_indices = torch.tensor(range(2*num_cdf_sample, 3*num_cdf_sample)).to(device) # dynamic noise

    def get_segment(self, y, indices):
        return torch.index_select(y, 1, indices)

    def mae(self, pred, target):
        
        yp_dir = self.get_segment(pred, self.dir_indices)
        yt_dir = self.get_segment(target, self.dir_indices)
        
        yp_cas = self.get_segment(pred, self.cas_indices)
        yt_cas = self.get_segment(target, self.cas_indices)
        
        yp_dyn = self.get_segment(pred, self.dyn_indices)
        yt_dyn = self.get_segment(target, self.dyn_indices)
        
        return (torch.abs(yp_dir - yt_dir).mean() +
                torch.abs(yp_cas - yt_cas).mean() +
                torch.abs(yp_dyn - yt_dyn).mean()) / 3.0
        
    def mre(self, pred, target):
        
        yp_dir = self.get_segment(pred, self.dir_indices)
        yt_dir = self.get_segment(target, self.dir_indices)
        
        yp_cas = self.get_segment(pred, self.cas_indices)
        yt_cas = self.get_segment(target, self.cas_indices)
        
        yp_dyn = self.get_segment(pred, self.dyn_indices)
        yt_dyn = self.get_segment(target, self.dyn_indices)
        
        return ((torch.abs(yp_dir - yt_dir) / yt_dir.abs()).mean() +
                (torch.abs(yp_cas - yt_cas) / yt_cas.abs()).mean() +
                (torch.abs(yp_dyn - yt_dyn) / yt_dyn.abs()).mean()) / 3.0

    def area_between_curves(self, pred, target):
        
        yp_dir = self.get_segment(pred, self.dir_indices)
        yt_dir = self.get_segment(target, self.dir_indices)
        # mask = (yt_dir != self.mask_val).float()
        diff_dir = torch.abs(yp_dir - yt_dir)
        area_dir = torch.trapz(diff_dir, dim=1)
        
        yp_cas = self.get_segment(pred, self.cas_indices)
        yt_cas = self.get_segment(target, self.cas_indices)
        # mask = (yt_cas != self.mask_val).float()
        diff_cas = torch.abs(yp_cas - yt_cas)
        area_cas = torch.trapz(diff_cas, dim=1)
        
        yp_dyn = self.get_segment(pred, self.dyn_indices)
        yt_dyn = self.get_segment(target, self.dyn_indices)
        # mask = (yt_dyn != self.mask_val).float()
        diff_dyn = torch.abs(yp_dyn - yt_dyn)
        area_dyn = torch.trapz(diff_dyn, dim=1)
        
        return (area_dir.mean() +
                area_cas.mean() +
                area_dyn.mean()) / 3.0

    def ks_distance(self, pred, target):  
        
        yp_dir = self.get_segment(pred, self.dir_indices)
        yt_dir = self.get_segment(target, self.dir_indices)
        mask = (yt_dir != self.mask_val).float()
        diff_dir = torch.abs(yp_dir - yt_dir) * mask
        ks_dir = torch.max(diff_dir, dim=1).values
        
        yp_cas = self.get_segment(pred, self.cas_indices)
        yt_cas = self.get_segment(target, self.cas_indices)
        mask = (yt_cas != self.mask_val).float()
        diff_cas = torch.abs(yp_cas - yt_cas) * mask
        ks_cas = torch.max(diff_cas, dim=1).values
        
        yp_dyn = self.get_segment(pred, self.dyn_indices)
        yt_dyn = self.get_segment(target, self.dyn_indices)
        mask = (yt_dyn != self.mask_val).float()
        diff_dyn = torch.abs(yp_dyn - yt_dyn) * mask
        ks_dyn = torch.max(diff_dyn, dim=1).values

        return (ks_dir.mean() +
                ks_cas.mean() +
                ks_dyn.mean()) / 3.0

    def slope_mae(self, pred, target):
        
        yp_dir = self.get_segment(pred, self.dir_indices)
        yt_dir = self.get_segment(target, self.dir_indices)
        d_pred = yp_dir[:, 1:] - yp_dir[:, :-1]
        d_target = yt_dir[:, 1:] - yt_dir[:, :-1]
        mask = (yt_dir > self.mask_val + 1e-3).float()
        valid_pairs = mask[:, :-1] * mask[:, 1:]
        slope_dir = torch.abs(d_pred - d_target) * valid_pairs
        
        yp_cas = self.get_segment(pred, self.cas_indices)
        yt_cas = self.get_segment(target, self.cas_indices)
        d_pred = yp_cas[:, 1:] - yp_cas[:, :-1]
        d_target = yt_cas[:, 1:] - yt_cas[:, :-1]
        mask = (yt_cas > self.mask_val + 1e-3).float()
        valid_pairs = mask[:, :-1] * mask[:, 1:]
        slope_cas = torch.abs(d_pred - d_target) * valid_pairs
        
        yp_dyn = self.get_segment(pred, self.dyn_indices)
        yt_dyn = self.get_segment(target, self.dyn_indices)
        d_pred = yp_dyn[:, 1:] - yp_dyn[:, :-1]
        d_target = yt_dyn[:, 1:] - yt_dyn[:, :-1]
        mask = (yt_dyn > self.mask_val + 1e-3).float()
        valid_pairs = mask[:, :-1] * mask[:, 1:]
        slope_dyn = torch.abs(d_pred - d_target) * valid_pairs
        
        return (slope_dir.mean() +
                slope_cas.mean() +
                slope_dyn.mean()) / 3.0

    def evaluate(self, pred, target):
        return self.mae(pred, target).item(), \
                self.mre(pred, target).item(), \
                self.area_between_curves(pred, target).item(),  \
                self.ks_distance(pred, target).item(), \
                self.slope_mae(pred, target).item()


# Early Stopping Strategy
class earlyStopping():
    def __init__(self, patience=10, delta=0.):
        
        self.patience = patience
        self.delta = delta
        self.min_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        
        if val_loss < self.min_loss - self.delta:
            self.min_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                return True  # 返回 True 表示停止训练
        return False


# train vit_NN
class train_impl():
    def __init__(self, dataset, num_feat, num_sample, num_output,
                batch_size, num_epoch, model_path, loss_path,
                lr=1e-5, weight_decay=1e-5, patience=10):
        super().__init__()

        self.dataset = dataset
        self.num_feat = num_feat
        self.num_sample = num_sample
        self.num_output = num_output
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model_path = model_path
        self.loss_path = loss_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        
        self.invalid_value = -600.0
        
        self.bins = dataset.bins
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def K_fold_cross_validation(self, num_fold, loss_delta):

        val_loss_min = float('inf')  # 当前最优模型的loss
        best_model = None  # 最优模型
        results = []
        train_loss_set = []
        val_loss_set = []

        kfold = KFold(n_splits=num_fold, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):

            print(f"Fold {fold + 1} / {num_fold}")

            # step1 划分训练集和验证集
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            # step2 重新初始化模型，每一折从头开始训练
            model = vit_NN(self.bins, num_feat=self.num_feat)
            if torch.cuda.is_available():
                cudnn.benchmark = True
                model = model.to(self.device)

            # step3 创建损失函数
            loss_fn = composite_loss(device=self.device, num_cdf_sample=self.num_sample, delta=loss_delta)

            # step4 创建AdaW优化器
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # step5 创建调度器
            # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.3, last_epoch = -1)
            # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)

            # step6 创建早停策略
            early_stop = earlyStopping(patience=self.patience)

            # step7 训练模型
            train_loss_curve, val_loss_curve, best_model_per_fold, min_val_loss = self.train(model, train_loader, val_loader, loss_fn,
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
                print("Update best loss: {:.4f}".format(val_loss_min))
                torch.save(best_model, self.model_path)

        # step10 存模型和train&test loss曲线
        print("Best Loss: {:.4f}".format(val_loss_min))
        torch.save(best_model, self.model_path)
        with open(self.loss_path, 'w') as f:
            json.dump({'train_loss': train_loss_set, 
                       'val_loss': val_loss_set, 
                       'results': results}, f)

        # step11 绘出所有折的loss曲线
        for _, (train_loss_curve, val_loss_curve) in enumerate(zip(train_loss_set, val_loss_set)):
            self.plot_loss(train_loss_curve, val_loss_curve)

    def lr_lambda(self, epoch):

        first_stage = 20.0
        second_stage = 100.0
        top1_val = 3e-4
        top2_val = 1e-4
        diff_stage = second_stage - first_stage

        if epoch + 1 <= (int)(first_stage):
            return (top1_val / self.lr) * ((epoch + 1) / first_stage)
        elif (int)(first_stage) < epoch + 1 <= (int)(second_stage):
            decay_rate = top2_val/self.lr
            return decay_rate **( ((int)(second_stage)-epoch) / diff_stage)
        else:
            return 1.0

    def train(self, model, train_loader, val_loader, loss_fn, 
              optimizer, lr_scheduler, early_stop):

        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(self.num_epoch):
            print(f'Epoch: {epoch}')          
            train_loss, val_loss = self.fit_one_epoch(model, train_loader, val_loader,
                                                    loss_fn, optimizer, self.device)
            lr_scheduler.step()

            train_loss_vec.append(train_loss)
            val_loss_vec.append(val_loss)

            if early_stop(val_loss, model):
                break

        best_model = early_stop.best_model
        min_val_loss = early_stop.min_loss
        
        return train_loss_vec, val_loss_vec, best_model, min_val_loss

    def fit_one_epoch(self, model, train_loader, val_loader,
                    loss_fn, optimizer, device):

        start_time = time.time()

        model.train()
        train_loss_list = []
        train_data_size = len(train_loader)
        for step, data in enumerate(train_loader):
            feature, labels = data[0], data[1]
            feature, labels = feature.to(device), labels.to(device)
            true_mask = (labels != self.invalid_value).float()

            optimizer.zero_grad()  # 清零梯度
            pred_cdf, pred_mask = model(feature)  # 前向传播
            
            loss_value = loss_fn(pred_cdf, labels, pred_mask, true_mask)  # 计算loss值
            
            error = (pred_cdf - labels) * true_mask
            abs_error_mean = error.abs().sum().item() / true_mask.sum().item()

            loss_value.backward()  # 反向传播
            optimizer.step()  # 优化器迭代

            train_loss_list.append(loss_value.item())

            # 画进度条
            self.draw_process_bar(step, train_data_size, loss_value, abs_error_mean, 'train')

        print()
        print('train_loss:{:.3f}'.format(np.mean(train_loss_list)))

        model.eval()
        val_loss_list = []
        val_mae_list = []
        val_data_size = len(val_loader)
        for step, data in enumerate(val_loader):
            feature, labels = data[0], data[1]
            feature, labels = feature.to(device), labels.to(device)
            true_mask = (labels != self.invalid_value).float()

            with torch.no_grad():
                pred_cdf, pred_mask = model(feature)
                loss_value = loss_fn(pred_cdf, labels, pred_mask, true_mask)

                error = (pred_cdf - labels) * true_mask
                abs_error_mean = error.abs().sum().item() / true_mask.sum().item()

                val_loss_list.append(loss_value.item())
                val_mae_list.append(abs_error_mean)

            self.draw_process_bar(step, val_data_size, loss_value, abs_error_mean, 'validate')

        print()
        print('validate_loss:{:.3f}, validate_mae:{:.3f}, epoch_time:{:.3f} s'.format(
            np.mean(val_loss_list), 
            np.mean(val_mae_list),
            time.time()-start_time))

        return np.mean(train_loss_list), np.mean(val_loss_list)

    def draw_process_bar(self, step, length, loss_value, abs_error_mean, phase):

        rate = (step + 1) / length
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r{:s} loss: {:^3.0f}%[{}->{}]{:.3f}, {:.4f}".format(
            phase, int(rate * 100), a, b, loss_value, abs_error_mean), end="")

    def plot_loss(self, train_loss_vec, test_loss_vec):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss_vec, 'r', label='training loss')
        ax.plot(test_loss_vec, 'b--', label='validation loss')
        ax.set_title(f'Training and Validation Loss of LPS-Net')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()
        plt.close(fig)


class train():
    def __init__(self, train_dataset, num_feat, num_sample, num_output, 
                        batch_size, num_epoch, model_path, loss_path):

        self.train_impl = train_impl(train_dataset, num_feat, num_sample, num_output,
                                     batch_size, num_epoch, model_path, loss_path)

    def K_fold_cross_validation(self, num_fold, loss_delta):
        
        self.train_impl.K_fold_cross_validation(num_fold=num_fold, loss_delta=loss_delta)


# test vit_NN
class test_impl():
    def __init__(self, test_data_path, train_dataset, num_feat, 
                 num_sample, num_output, batch_size, load_path):
        super(test_impl, self).__init__()

        test_dataset = util_dataset.test_vit_dataset(test_data_path, num_output)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.test_model = vit_NN(train_dataset.bins, num_feat=num_feat)
        self.num_feat = num_feat
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.load_path = load_path
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = composite_loss(device=self.device)
        self.patcher = CDFPatcher(
                            max_invalid_len=18,
                            fill_method='linear',
                            fallback_value=-600.0
                        )
        self.invalid_value = -600.0
        self.evaluator = CdfEvaluator(self.device)

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
        feature, labels = feature.to(self.device), labels.to(self.device)
        true_mask = (labels != self.invalid_value).float()
        
        with torch.no_grad():
            pred_cdf, pred_mask = test_model(feature)
            outputs = self.inference_process(pred_cdf, pred_mask, threshold=0.5)
            loss = self.loss_fn(outputs, labels, pred_mask, true_mask)
            mae, mre, abc, ks, Slope_MAE = self.evaluator.evaluate(outputs, labels)
            
            # mae = torch.mean((outputs - labels).abs()).item()  # MAE
            # mre = torch.mean((outputs - labels).abs() / labels.abs()).item()  # MRE
            return loss.item(), mae, mre, abc, ks, Slope_MAE

    def inference_process(self, pred_cdf, pred_mask, threshold=0.5):
        valid_mask = (pred_mask >= threshold).float()  # [B, 63]，1表示有效点
        processed = pred_cdf * valid_mask + self.invalid_value * (1.0 - valid_mask)
        return processed

    def test(self, plot_cdf=False):

        test_model = self.test_model
        test_model.to(self.device)
        test_model.load_state_dict(torch.load(self.load_path, weights_only=True))

        test_model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_abc = 0.0
        total_ks = 0.0
        total_Slope_MAE = 0.0
        total_mre = 0.0
        total_time = 0.0
        count = 0
        skipped_batches = 0

        for batch_idx, data in enumerate(self.test_loader):

            if len(data) != 2:
                print("Invalid data format, skipping this batch.")
                skipped_batches += 1
                continue

            # test data and measure time
            (loss, mae, mre, abc, ks, Slope_MAE), elapsed_time = self.process_data(test_model, data)
            
            if loss == float('inf') or mae == float('inf')  or mre == float('inf') or abc == float('inf') or ks == float('inf') or Slope_MAE == float('inf'):
                print(f"Error occurred in batch {batch_idx}, skipping this batch.")
                skipped_batches += 1
                continue

            total_loss += loss
            total_mae += mae
            total_abc += abc
            total_ks += ks
            total_Slope_MAE += Slope_MAE
            total_mre += mre
            total_time += elapsed_time
            count += 1
            
            # visualize curves of cdf
            if plot_cdf:
                feature, labels = data[0].to(self.device), data[1].to(self.device)
                pred_cdf, pred_mask = test_model(feature)
                outputs = self.inference_process(pred_cdf, pred_mask, threshold=0.5)
                self.plot_cdf_curve(labels, outputs)

        # prevent divide zero
        if count == 0:
            print("No valid data batches found.")
            return

        mean_time = total_time / count
        mean_loss = total_loss / count
        mean_mae = total_mae / count
        mean_mre = total_mre / count
        mean_abc = total_abc / count
        mean_ks = total_ks / count
        mean_Slope_MAE = total_Slope_MAE / count

        print('Avg calculate time: {:.4f}'.format(mean_time))
        print('Avg loss: {:.4f}'.format(mean_loss))
        print('MAE: {:.4f}'.format(mean_mae))
        print('MRE: {:.4f}'.format(mean_mre))
        print('ABC: {:.4f}'.format(mean_abc))
        print('KS: {:.4f}'.format(mean_ks))
        print('Slope_MAE: {:.4f}'.format(mean_Slope_MAE))
        
    def plot_cdf_curve(self, labels, outputs):
        
        dir_indices = torch.tensor(range(0*self.num_sample,1*self.num_sample)).to(self.device)
        cas_indices = torch.tensor(range(1*self.num_sample,2*self.num_sample)).to(self.device)
        dyn_indices = torch.tensor(range(2*self.num_sample,3*self.num_sample)).to(self.device)

        y_label = np.linspace(0, 1, self.num_sample)
        
        dir_cdf_true = torch.index_select(labels, 1, dir_indices)
        cas_cdf_true = torch.index_select(labels, 1, cas_indices)
        dyn_cdf_true = torch.index_select(labels, 1, dyn_indices)

        dir_cdf_pred = torch.index_select(outputs, 1, dir_indices)
        cas_cdf_pred = torch.index_select(outputs, 1, cas_indices)
        dyn_cdf_pred = torch.index_select(outputs, 1, dyn_indices)
        
        plt.rcParams.update({'font.size': 40})
        plt.rcParams['font.family'] = 'Times New Roman'
        
        for elem in range(labels.shape[0]):

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(dir_cdf_pred.cpu().detach().numpy()[elem, :], y_label, 'r', label='predict')
            ax1.plot(dir_cdf_true.cpu().detach().numpy()[elem, :], y_label, 'r--', label='truth')
            ax1.set_xlabel('Signal Power (dBm)')
            ax1.set_ylabel('Probability')
            ax1.set_title('CDF of Direct Signal Power')
            ax1.legend()

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(cas_cdf_pred.cpu().detach().numpy()[elem, :], y_label, 'g', label='predict')
            ax2.plot(cas_cdf_true.cpu().detach().numpy()[elem, :], y_label, 'g--', label='truth')
            ax2.set_xlabel('Signal Power (dBm)')
            ax2.set_ylabel('Probability')
            ax2.set_title('CDF of Cascaded Signal Power')
            ax2.legend()

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            ax3.plot(dyn_cdf_pred.cpu().detach().numpy()[elem, :], y_label, 'b', label='predict')
            ax3.plot(dyn_cdf_true.cpu().detach().numpy()[elem, :], y_label, 'b--', label='truth')
            ax3.set_xlabel('Dynamic Noise Power (dBm)')
            ax3.set_ylabel('Probability')
            ax3.set_title('CDF of Cascaded Dynamic Noise Power')
            ax3.legend()
            
            plt.show()
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

    def plot_loss_curve(self, loss_set):

        plt.rcParams.update({'font.size': 15})
        
        fig1 = plt.figure()
        ax4 = fig1.add_subplot(111)
        ax4.plot(range(len(loss_set)), loss_set, 'r')
        ax4.set_xlabel('Sample')
        ax4.set_ylabel('Loss')
        ax4.set_title('Test Loss of LPS-Net')
        plt.show()
        plt.close(fig1)


class test():
    def __init__(self, test_data_path, train_dataset, num_feat, 
                 num_sample, num_output, batch_size, load_path):

        self.test_impl = test_impl(test_data_path, train_dataset, num_feat, 
                            num_sample, num_output, batch_size, load_path)

    def test(self, plot_cdf=False):

        self.test_impl.test(plot_cdf=plot_cdf)


# calculate vit_NN
class cal_impl():
    def __init__(self, cal_data_path, train_dataset, num_feat, 
                 num_sample, num_output, batch_size, load_path):
        super(cal_impl, self).__init__()

        cal_dataset = util_dataset.cal_vit_dataset(cal_data_path)
        self.cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)
        self.model = vit_NN(train_dataset.bins, num_feat=num_feat)
        self.num_feat = num_feat
        self.num_sample = num_sample
        self.num_output = num_output
        self.batch_size = batch_size
        self.load_path = load_path
        
        self.invalid_value = -600.0
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = composite_loss(device=self.device)

    def cal(self, plot_cdf=False):

        cal_model = self.model
        cal_model.to(self.device)
        cal_model.load_state_dict(torch.load(self.load_path, weights_only=True))

        # test
        cal_model.eval()

        result = torch.empty(0, dtype=torch.float32, device=self.device)
        for _, feature in enumerate(self.cal_loader):

            with torch.no_grad():
                feature = feature.to(self.device)
                pred_cdf, pred_mask = cal_model(feature)  # 前向传播
                outputs = self.inference_process(pred_cdf, pred_mask, threshold=0.5)
                result = torch.cat((result, outputs), dim=0)

            # visualize curves of cdf
            if plot_cdf:
                self.plot_cdf_curve(outputs)
        
        return result.cpu().numpy()
    
    def inference_process(self, pred_cdf, pred_mask, threshold=0.5):
        valid_mask = (pred_mask >= threshold).float()
        processed = pred_cdf * valid_mask + self.invalid_value * (1.0 - valid_mask)
        return processed

    def plot_cdf_curve(self, outputs):

        dir_indices = torch.tensor(range(0*self.num_sample,1*self.num_sample)).to(self.device)
        cas_indices = torch.tensor(range(1*self.num_sample,2*self.num_sample)).to(self.device)
        dyn_indices = torch.tensor(range(2*self.num_sample,3*self.num_sample)).to(self.device)

        y_label = np.linspace(0, 1, self.num_sample)

        dir_cdf_pred = torch.index_select(outputs, 1, dir_indices)
        cas_cdf_pred = torch.index_select(outputs, 1, cas_indices)
        dyn_cdf_pred = torch.index_select(outputs, 1, dyn_indices)

        plt.rcParams.update({'font.size': 15})

        for elem in range(outputs.shape[0]):

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(131)
            ax1.plot(dir_cdf_pred.cpu().detach().numpy()[elem, :], y_label, 'r', label='predict')
            ax1.set_xlabel('Signal Power (dBm)')
            ax1.set_ylabel('Probability')
            ax1.set_title('CDF of Direct Signal Power')
            ax1.legend()

            ax2 = fig1.add_subplot(132)
            ax2.plot(cas_cdf_pred.cpu().detach().numpy()[elem, :], y_label, 'g', label='predict')
            ax2.set_xlabel('Signal Power (dBm)')
            ax2.set_ylabel('Probability')
            ax2.set_title('CDF of Cascaded Signal Power')
            ax2.legend()

            ax3 = fig1.add_subplot(133)
            ax3.plot(dyn_cdf_pred.cpu().detach().numpy()[elem, :], y_label, 'b', label='predict')
            ax3.set_xlabel('Dynamic Noise Power (dBm)')
            ax3.set_ylabel('Probability')
            ax3.set_title('CDF of Cascaded Dynamic Noise Power')
            ax3.legend()

            plt.show()
            plt.close(fig1)

    def save(self, data, file_name='data', data_type='mat'):
        data_mat = {"cdf_data": data}
        savemat(f"{file_name}.{data_type}", data_mat)


class cal():
    def __init__(self, cal_data_path, train_dataset, num_feat, 
                 num_sample, num_output, batch_size, load_path):

        self.cal_impl = cal_impl(cal_data_path, train_dataset, num_feat, 
                            num_sample, num_output, batch_size, load_path)

    def cal(self, plot_cdf=False):

        return self.cal_impl.cal(plot_cdf=plot_cdf)

    def save(self, data, file_name='data', data_type='mat'):
        self.cal_impl.save(data, file_name, data_type)
