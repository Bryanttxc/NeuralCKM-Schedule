import torch
import torch.nn as nn
import torch.nn.functional as func
from utils.utils import to_2tuple
from third.rtdl_num_embeddings.package.rtdl_num_embeddings import PiecewiseLinearEmbeddings

###### 定义UESENet Framework ######

# NN-1 output module
class MLP_head(nn.Module):
    def __init__(self, num_classes, cdf_dim, embed_dim):
        super(MLP_head, self).__init__()
        self.num_classes = num_classes
        self.cdf_dim = cdf_dim
        
        # create an FC for each class
        self.fcs = nn.ModuleList([nn.Sequential(
                        nn.Linear(embed_dim, embed_dim), 
                        nn.GELU(),
                        nn.Linear(embed_dim, 1))
                        for _ in range(num_classes)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_classes):
            # divide inputs into num_classes parts
            # dim: [batch_size, cdf_dim, embed_dim]
            xi = x[:, i*self.cdf_dim:(i+1)*self.cdf_dim, :]
            output_i = self.fcs[i](xi)
            outputs.append(output_i.squeeze(dim=2))

        # concatenate outputs in dim: [batch_size, num_output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


# NN-1 mask module
class MASK_head(nn.Module):
    def __init__(self, num_classes, cdf_dim, embed_dim):
        super(MASK_head, self).__init__()

        self.num_classes = num_classes
        self.cdf_dim = cdf_dim
        self.fcs = nn.ModuleList([nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.ReLU(),
                        nn.Linear(embed_dim, 1),
                        nn.Sigmoid())
                        for _ in range(num_classes)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_classes):
            # divide inputs into num_classes parts
            # dim: [batch_size, cdf_dim, embed_dim]
            xi = x[:, i*self.cdf_dim:(i+1)*self.cdf_dim, :]
            output_i = self.fcs[i](xi)
            outputs.append(output_i.squeeze(dim=2))

        # concatenate outputs in dim: [batch_size, num_output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


# NN-2 position embedding
def position_embedding(init_pos_embed, feat_dim):
    num_IRS = (feat_dim - 2) // 2
    first_three = init_pos_embed[:, :3, :]  # shape: (1, 3, 256)
    fourth = init_pos_embed[:, 3:4, :]  # shape: (1, 1, 256)
    fifth = init_pos_embed[:, 4:5, :]   # shape: (1, 1, 256)
    
    # repeat
    fourth_repeated = fourth.repeat(1, num_IRS, 1)  # (1, I-1, 256)
    fifth_repeated = fifth.repeat(1, num_IRS, 1)    # (1, I, 256)

    final_encoding = torch.cat([first_three, fourth_repeated, fifth_repeated], dim=1)  # (1, 2I+2, 256)
    return final_encoding


# class MLP_head(nn.Module):
#     def __init__(self, num_classes, cdf_dim, embed_dim, type_embed_dim=32):
#         """
#         num_classes: 类别数（例如3：direct, cascade, dynamic）
#         cdf_dim: 每类的CDF点数（例如21）
#         embed_dim: 输入token的维度（例如256）
#         type_embed_dim: 类别embedding的维度（可调）
#         """
#         super(MLP_head, self).__init__()

#         self.num_classes = num_classes
#         self.cdf_dim = cdf_dim
#         self.embed_dim = embed_dim
#         self.total_points = num_classes * cdf_dim

#         # 类别embedding：num_classes个type，每个有type_embed_dim维
#         self.type_embedding = nn.Embedding(num_classes, type_embed_dim)

#         # 定义共享的 MLP
#         self.shared_mlp = nn.Sequential(
#             nn.Linear(embed_dim + type_embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, 1)
#         )

#     def forward(self, x):
#         """
#         x: shape [B, total_output_points, embed_dim]
#         返回: [B, total_output_points]
#         """
#         B, N, D = x.shape
#         assert N == self.total_points

#         outputs = []
#         for i in range(self.num_classes):
#             # 当前类别的token序列 (B, cdf_dim, embed_dim)
#             xi = x[:, i*self.cdf_dim:(i+1)*self.cdf_dim, :]

#             # 类别embedding: shape (1, cdf_dim, type_embed_dim)
#             type_id = torch.tensor(i, dtype=torch.long, device=x.device)
#             type_embed = self.type_embedding(type_id).unsqueeze(0).unsqueeze(0)  # (1, 1, type_embed_dim)
#             type_embed = type_embed.expand(B, self.cdf_dim, -1)  # (B, cdf_dim, type_embed_dim)

#             # 拼接输入
#             xi_with_type = torch.cat([xi, type_embed], dim=2)  # (B, cdf_dim, embed_dim + type_embed_dim)

#             # 喂入共享MLP
#             out_i = self.shared_mlp(xi_with_type).squeeze(dim=2)  # (B, cdf_dim)

#             outputs.append(out_i)

#         # 拼接所有类别输出
#         outputs = torch.cat(outputs, dim=1)  # (B, total_output_points)
#         return outputs


# # cdf Position Encoding
# def position_encoding(feat_dim, effec_dim, embed_dim):
#     num_IRS = (feat_dim - 2) // 2
    
#     embed_specs = [
#             (1, 2.0),  # direct link
#             (1, 3.0),  # cascade link
#             (num_IRS, 4.0),  # scatter link
#             (num_IRS, 5.0)  # dynamic noise (包含所有IRS)
#     ]
    
#     # 使用列表推导式生成所有嵌入部分
#     pos_cls = torch.ones(1, embed_dim) * 1.0
#     pos_embed = torch.cat([
#         torch.ones(size, embed_dim) * val
#         for size, val in embed_specs], dim=0)
#     pos_embed = torch.cat((pos_cls, pos_embed), dim = 0).unsqueeze(0) # 1 2 3 4 5

#     return pos_embed


# multi-head self-attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 多头注意力机制的头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = head_dim ** -0.5  # 归一化参数

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)  # 产生qkv
        self.attn_drop = nn.Dropout(attn_drop)  # attention_score的dropout
        self.proj = nn.Linear(dim, dim)  # 多头注意力合并之后的语义空间转化
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout

    def forward(self, x):
        B, N, C = x.shape  # bach_size的大小，sequence的长度， 每个token的维度

        # (B, N, C) -> (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 单独取出q, k, v
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)  # 获取归一化后的attention_score
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Encoder里的MLP
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, drop = 0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层全连接层
        self.act = nn.GELU()  # 激活函数 GELU() -> 原来
        self.drop1 = nn.Dropout(drop_probs[0])  # 随机dropout
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层全连接层
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# Transformer encoder block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4., qkv_bias = False, drop = 0., attn_drop = 0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)  # 对输入进行layernorm处理
        self.attn = Attention(dim, num_heads = num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)
        self.norm2 = nn.LayerNorm(dim)  # 对self-attention之后的结果进行layernorm处理
        mlp_hidden_dim = int(dim * mlp_ratio)  # feedforward网络中间层维度
        self.mlp = MLP(in_features = dim, hidden_features = mlp_hidden_dim, drop = drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # 残差结构
        x = x + self.mlp(self.norm2(x))
        return x


# NN-1 framework
class VitTransformer(nn.Module):
    def __init__(self, bins, cdf_dim = 21, input_dim = 15, output_dim = 63, 
                 embed_dim = 256, depth = 4, num_heads = 4, mlp_ratio = 4, 
                 qkv_bias = False, drop_rate = 0., attn_drop_rate = 0.):
        super(VitTransformer, self).__init__()
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
        self.pos_drop = nn.Dropout(drop_rate)
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, 
                  drop = drop_rate, attn_drop = attn_drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # MLP head
        self.head = MLP_head(num_classes=output_dim//cdf_dim, cdf_dim=cdf_dim, embed_dim=embed_dim)
        # # mask head
        self.mask_head = MASK_head(num_classes=output_dim//cdf_dim, cdf_dim=cdf_dim, embed_dim=embed_dim)
        # self.mask_head = nn.Sequential(
        #                     nn.Linear(embed_dim, embed_dim),
        #                     nn.ReLU(),
        #                     nn.Linear(embed_dim, 1),
        #                     nn.Sigmoid())

    def forward(self, x):
        B = x.shape[0]  # batch_size

        x = self.embedding(x)  # dim: batch_size x feat_dim x embed_dim
        cls_token = self.cls_token.expand(B, -1, -1)  # dim: batch_size x output_dim x embed_dim
        x = torch.cat((cls_token, x), dim = 1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        x = x[:, 0:self.output_dim]
        pred_cdf = self.head(x)
        pred_mask = self.mask_head(x).squeeze(-1)
        return pred_cdf, pred_mask


# NN-2 framework
class CdfTransformer(nn.Module):
    def __init__(self, bins, device, cdf_dim = 21, output_dim = 1, embed_dim = 256, 
                 depth = 8, num_heads = 4, mlp_ratio = 4, qkv_bias = False,
                 drop_rate = 0., attn_drop_rate = 0.):
        super(CdfTransformer, self).__init__()
        self.device = device
        self.cdf_dim = cdf_dim  # cdf维度数
        self.embed_dim = embed_dim  # 每个token的维度数
        self.output_dim = output_dim  # 输出指标数
        
        # PLE
        self.PLE_dim = embed_dim // cdf_dim
        self.embedding = PiecewiseLinearEmbeddings(bins, d_embedding=self.PLE_dim, activation=True, version='A')
        self.pad_dim = self.embed_dim - self.PLE_dim * self.cdf_dim
        # target token
        self.cls_token = nn.Parameter(torch.zeros(1, output_dim, embed_dim))
        # position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 5, embed_dim))
        # dropout
        self.pos_drop = nn.Dropout(drop_rate)
        # Transformer block
        self.blocks = nn.Sequential(*[
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, 
                  drop = drop_rate, attn_drop = attn_drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # output
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim))

    def forward(self, x):
        B = x.shape[0]
        F = x.shape[1]

        # feature embedding
        x_embed = torch.empty(0, dtype=torch.float32, device=self.device)
        for i in range(F):
            tmp = self.embedding(x[:, i, :])
            tmp = tmp.reshape(B, 1, -1)
            x_embed = torch.cat((x_embed, func.pad(tmp, (0, self.pad_dim))), dim=1)
        x = x_embed

        # x = self.embedding(x)  # batch_size x feat_dim x embed_dim
        cls_token = self.cls_token.expand(B, -1, -1)  # batch_size x output_dim x embed_dim
        x = torch.cat((cls_token, x), dim = 1)  # batch_size x (feat_dim + output_dim) x embed_dim

        # position embedding
        # pos_embed = position_encoding(feat_dim, self.embed_dim - self.pad_dim, self.embed_dim).to(self.device)
        pos_embed = position_embedding(self.pos_embedding, F)
        x = self.pos_drop(x + pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        x = x[:,0]
        x = self.head(x)
        return x


# NN-1 entry
def vit_NN(bins, num_feat):
    model = VitTransformer(bins, cdf_dim = 21, input_dim = num_feat, output_dim = 63,
                              embed_dim = 256, depth = 4, num_heads = 4, mlp_ratio = 4, 
                              qkv_bias = True, drop_rate = 0., attn_drop_rate = 0.)
    return model


# NN-2 entry
def cdf_NN(bins, device, cdf_dim = 16):
    model = CdfTransformer(bins, device, cdf_dim = cdf_dim, output_dim = 1, 
                           embed_dim = 256, depth = 4, num_heads = 4, mlp_ratio = 4, 
                           qkv_bias = True, drop_rate = 0., attn_drop_rate = 0.)
    return model
