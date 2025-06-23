import torch
from third.rtdl_num_embeddings.package.rtdl_num_embeddings import compute_bins

import utils.NN_1 as vit
import utils.NN_2 as cdf
import baseline.MLP as mlp
import baseline.LSTM as lstm
import utils.dataloader as util_dataset

###### UESENet entry function ######

class vit_entry():
    def __init__(self, train_data_path, num_feat, num_sample,
                 num_output, batch_size, num_epoch, num_fold,
                 loss_delta):
        self.num_feat = num_feat
        self.num_sample = num_sample
        self.num_output = num_output
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_fold = num_fold
        self.loss_delta = loss_delta
        self.train_dataset = util_dataset.train_vit_dataset(train_data_path, num_output)

    def train(self, model_path, loss_path, num_fold=None):
        train_obj = vit.train(self.train_dataset, self.num_feat, self.num_sample, self.num_output,
                                self.batch_size, self.num_epoch, model_path, loss_path)
        num_fold = num_fold or self.num_fold
        train_obj.K_fold_cross_validation(num_fold, self.loss_delta)

    def test(self, test_data_path, model_path, is_plot=False):
        test_obj = vit.test(test_data_path, self.train_dataset, self.num_feat, self.num_sample, 
                            self.num_output, self.batch_size, model_path)
        test_obj.test(plot_cdf=is_plot)

    def cal(self, cal_data_path, model_path, save_path):
        cal_obj = vit.cal(cal_data_path, self.train_dataset, self.num_feat, self.num_sample, 
                          self.num_output, self.batch_size, model_path)
        result = cal_obj.cal(plot_cdf=False)
        cal_obj.save(result, save_path, 'mat')


class cdf_entry():
    def __init__(self, train_data_path, num_case, num_sample, num_output, batch_size, num_epoch, num_fold):
        self.num_case = num_case
        self.num_sample = num_sample
        self.num_output = num_output
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_fold = num_fold
        
        train_dataset = []
        whole_data = torch.empty(0, dtype=torch.float32)
        for data_path in train_data_path:
            train_data = util_dataset.train_cdf_dataset(data_path, num_output, 
                                    cdf_dim=num_sample, var_name='train_result')
            whole_data = torch.cat((whole_data, train_data.total_feat), dim=0)
            train_dataset.append(train_data)

        self.train_dataset = train_dataset
        self.bins = compute_bins(whole_data)

    def train(self, model_path, loss_path, num_fold=None):
        train_obj = cdf.train(self.train_dataset, self.bins, self.num_case, self.num_sample, self.num_output,
                    self.batch_size, self.num_epoch, model_path, loss_path)
        num_fold = num_fold or self.num_fold
        train_obj.K_fold_cross_validation(num_fold)

    def test(self, test_data_path, model_path, is_plot=False):
        test_obj = cdf.test(self.bins, test_data_path, self.num_case, self.num_sample,
                    self.num_output, self.batch_size, model_path)
        test_obj.test(plot_loss=is_plot)

    def cal(self, cal_data_path, model_path, save_path):
        cal_obj = cdf.cal(self.bins, cal_data_path, self.num_case, self.num_sample,
                    self.num_output, self.batch_size, model_path)
        result = cal_obj.cal(plot_SE=False)
        cal_obj.save(result, save_path, 'mat')


class mlp_entry():
    def __init__(self, train_data_path, num_feat, num_sample, num_output, batch_size, num_epoch, num_fold):
        self.num_feat = num_feat
        self.num_sample = num_sample
        self.num_output = num_output
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_fold = num_fold
        self.train_dataset = util_dataset.train_vit_dataset(train_data_path, num_output)

    def train(self, model_path, loss_path, num_fold=None):
        train_obj = mlp.train(self.train_dataset, self.num_feat, self.num_sample, self.num_output,
                    self.batch_size, self.num_epoch, model_path, loss_path)
        num_fold = num_fold or self.num_fold
        train_obj.K_fold_cross_validation(num_fold)

    def test(self, test_data_path, model_path, is_plot=False):
        test_obj = mlp.test(test_data_path, self.train_dataset, self.num_feat, 
                    self.num_sample, self.num_output, self.batch_size, model_path)
        test_obj.test(plot_cdf=is_plot)


class lstm_entry():
    def __init__(self, train_data_path, num_case, num_sample, num_output, batch_size, num_epoch, num_fold):
        self.num_case = num_case
        self.num_sample = num_sample
        self.num_output = num_output
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_fold = num_fold
        
        train_dataset = []
        whole_data = torch.empty(0, dtype=torch.float32)
        for data_path in train_data_path:
            train_data = util_dataset.train_cdf_dataset(data_path, num_output, 
                                    cdf_dim=num_sample, var_name='train_result')
            whole_data = torch.cat((whole_data, train_data.total_feat), dim=0)
            train_dataset.append(train_data)

        self.train_dataset = train_dataset
        self.bins = compute_bins(whole_data)

    def train(self, model_path, loss_path, num_fold=None):
        train_obj = lstm.train(self.train_dataset, self.bins, self.num_case, self.num_sample, self.num_output,
                    self.batch_size, self.num_epoch, model_path, loss_path)
        num_fold = num_fold or self.num_fold
        train_obj.K_fold_cross_validation(num_fold)

    def test(self, test_data_path, model_path, is_plot=False):
        test_obj = lstm.test(self.bins, test_data_path, self.num_case, self.num_sample,
                    self.num_output, self.batch_size, model_path)
        test_obj.test(is_plot)
