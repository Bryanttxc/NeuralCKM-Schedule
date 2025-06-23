import torch
from torch.utils.data import Dataset
import scipy.io
import pandas as pd
from third.rtdl_num_embeddings.package.rtdl_num_embeddings import compute_bins

###### data-loading ######

class UESE_dataset(Dataset):
    def __init__(self, file_type, num_output = 63, cdf_dim = 16, var_name = '', train = True):
        super(UESE_dataset, self).__init__()

        # load data
        if '.xlsx' in file_type:
            in_file = pd.read_excel(file_type)
            in_file = in_file.to_numpy()

            # create bins if train_set
            if train:
                total_feat = in_file[:, :-num_output]
                total_feat = torch.from_numpy(total_feat).float()
                self.bins = compute_bins(total_feat)

        elif '.mat' in file_type:
            in_file = scipy.io.loadmat(file_type)
            in_file = in_file[var_name]

            if train:
                total_feat = in_file[:, :-num_output]
                total_feat = total_feat.reshape(-1, 1).reshape(-1, cdf_dim)
                self.total_feat = torch.from_numpy(total_feat).float()

        # connect features and labels of data by tuple
        samples = []
        for idx in range(0, len(in_file)):
            feat = in_file[idx, :-num_output]  # dim: 1 x num_feat
            label = in_file[idx, -num_output:]  # dim: 1 x num_output
            feat, label = torch.from_numpy(feat).float(), torch.from_numpy(label).float()
            samples.append((feat, label))
        self.samples = samples

    def __getitem__(self, index):
        feat, label = self.samples[index]
        return feat, label

    def __len__(self):
        return len(self.samples)


class Cal_UESE_dataset(Dataset):
    def __init__(self, file_type, var_name=''):
        super(Cal_UESE_dataset, self).__init__()

        # load data
        if '.xlsx' in file_type:
            in_file = pd.read_excel(file_type)
            in_file = in_file.to_numpy()
            
        elif '.mat' in  file_type:
            in_file = scipy.io.loadmat(file_type)
            in_file = in_file[var_name]
        
        # connect features and labels of data by tuple
        samples = []
        for idx in range(0, len(in_file)):
            feat = in_file[idx, :]  # dim: 1 x num_feat
            feat = torch.from_numpy(feat).float()
            samples.append(feat)
        self.samples = samples

    def __getitem__(self, index):
        feat = self.samples[index]
        return feat

    def __len__(self):
        return len(self.samples)


def train_vit_dataset(file_type, num_output):
    return UESE_dataset(file_type, num_output=num_output, train=True)

def test_vit_dataset(file_type, num_output):
    return UESE_dataset(file_type, num_output=num_output, train=False)

def cal_vit_dataset(file_type, var_name = 'cal_result'):
    return Cal_UESE_dataset(file_type, var_name=var_name)

def train_cdf_dataset(file_type, num_output, cdf_dim = 16, var_name = 'train_result'):
    return UESE_dataset(file_type, num_output=num_output, cdf_dim=cdf_dim, var_name=var_name, train=True)

def test_cdf_dataset(file_type, num_output, cdf_dim = 16, var_name = 'test_result'):
    return UESE_dataset(file_type, num_output=num_output, cdf_dim=cdf_dim, var_name=var_name, train=False)

def cal_cdf_dataset(file_type, var_name = 'cal_result'):
    return Cal_UESE_dataset(file_type, var_name=var_name)
