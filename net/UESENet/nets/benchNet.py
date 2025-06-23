import torch
import torch.nn as nn
import torch.nn.functional as func
from utils.utils import to_2tuple
from third.rtdl_num_embeddings.package.rtdl_num_embeddings import PiecewiseLinearEmbeddings

from nets.net import MLP

###### 定义benchmark framework ######

# simpMLP output module
class MLP_head(nn.Module):
    def __init__(self, num_classes, cdf_dim, embed_dim):
        super(MLP_head, self).__init__()

        self.num_classes = num_classes
        self.fcs = nn.ModuleList([nn.Sequential(
                        nn.Linear(embed_dim, embed_dim), 
                        nn.GELU(),
                        nn.Linear(embed_dim, cdf_dim))
                        for _ in range(num_classes)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_classes):
            # x dim: [batch_size, embed_dim]
            output_i = self.fcs[i](x)
            outputs.append(output_i)

        # Cascade outputs in dim: [batch_size, num_output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


# simpMLP mask module
class MASK_head(nn.Module):
    def __init__(self, num_classes, cdf_dim, embed_dim):
        super(MASK_head, self).__init__()

        self.num_classes = num_classes
        self.fcs = nn.ModuleList([nn.Sequential(
                        nn.Linear(embed_dim, embed_dim), 
                        nn.ReLU(),
                        nn.Linear(embed_dim, cdf_dim),
                        nn.Sigmoid())
                        for _ in range(num_classes)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_classes):
            # x dim: [batch_size, embed_dim]
            output_i = self.fcs[i](x)
            outputs.append(output_i)

        # Cascade outputs in dim: [batch_size, num_output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class simpMLP(nn.Module):
    def __init__(self, bins, cdf_dim = 21, input_dim = 15, output_dim = 63, 
                 embed_dim = 256, mlp_ratio = 4., drop = 0.):
        super(simpMLP, self).__init__()
        self.output_dim = output_dim  # 输出指标数
        self.input_dim = input_dim  # 输入特征数
        self.embed_dim = embed_dim  # 每个token的维度数

        # PLE
        self.embedding = PiecewiseLinearEmbeddings(bins, d_embedding=embed_dim, activation=True, version='A')
        # target_token
        self.cls_token = nn.Parameter(torch.zeros(1, output_dim, embed_dim))
        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, output_dim + input_dim, embed_dim))
        # dropout
        self.pos_drop = nn.Dropout(drop)
        
        self.norm = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=input_dim*embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim, drop=drop)

        self.head = MLP_head(num_classes=output_dim//cdf_dim, cdf_dim=cdf_dim, embed_dim=embed_dim)

        self.mask_head = MASK_head(num_classes=output_dim//cdf_dim, cdf_dim=cdf_dim, embed_dim=embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.norm(x)
        pred_cdf = self.head(x)
        pred_mask = self.mask_head(x)
        
        # x = self.norm(x)
        # x = x + self.mlp(x)
        # x = self.norm(x)
        
        # x = x[:, 0:self.output_dim]
        # pred_cdf = self.head(x)
        # pred_mask = self.mask_head(x).squeeze(-1)
        return pred_cdf, pred_mask


class LSTM(nn.Module):
    def __init__(self, bins, device, cdf_dim = 16, output_dim = 1, 
                 embed_dim = 256, num_layers = 1, drop = 0.):
        super(LSTM, self).__init__()
        
        self.device = device
        self.cdf_dim = cdf_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # embedding层
        self.PLE_dim = embed_dim // cdf_dim
        self.embedding = PiecewiseLinearEmbeddings(bins, d_embedding=self.PLE_dim, activation=True, version='A')
        self.pad_dim = self.embed_dim - self.PLE_dim * self.cdf_dim
        
        # LSTM层，batch_first=True使得输入数据的格式为 (batch_size, sequence_length, features)
        hidden_dim = embed_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=drop)

        # MLP output
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        feat_dim = x.shape[1]
        
         # feature embedding
        x_embed = torch.empty(0, dtype=torch.float32, device=self.device)
        for i in range(feat_dim):
            tmp = self.embedding(x[:, i, :])
            tmp = tmp.reshape(batch_size, 1, -1)
            x_embed = torch.cat((x_embed, func.pad(tmp, (0, self.pad_dim))), dim=1)
        x = x_embed
        
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :] # 取最后一个时间步的输出
        output = self.fc(out)
        
        return output


def simpMLP_NN(bins, num_feat):
    model = simpMLP(bins, cdf_dim = 21, input_dim = num_feat, output_dim = 63, 
                    embed_dim = 256, mlp_ratio = 4., drop = 0.)
    return model


def LSTM_NN(bins, device, cdf_dim = 16):
    model = LSTM(bins, device, cdf_dim = cdf_dim, output_dim = 1, 
                embed_dim = 256, num_layers = 1, drop = 0.)
    return model
