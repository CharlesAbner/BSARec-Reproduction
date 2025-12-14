# -*- coding: UTF-8 -*-
""" BSARec
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers
import torch.nn.functional as F
class PositionwiseFeedForward(nn.Module):
    "实现 FFN (Position-wise Feed-Forward) "
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 这种实现方式 (x -> w_1 -> gelu -> dropout -> w_2)
        # BSARecBlock
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

class BSARecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Weight for attentive inductive bias.')
        parser.add_argument('--c', type=int, default=5,
                            help='Frequency cutoff threshold.')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads

        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.args=args
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([ BSARecBlock(self.args) for _ in range(self.num_layers) ])

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()


        device = his_vectors.device
        his_vector = his_vectors[torch.arange(batch_size,device=device), lengths - 1, :]
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)

        u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
        i_v = i_vectors

        return {'prediction': prediction.view(batch_size, -1), 'u_v': u_v, 'i_v':i_v,
                'his_vector': his_vector}


class BSARec(SequentialModel, BSARecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'alpha', 'c'] # 加上了 alpha 和 c

    @staticmethod
    def parse_model_args(parser):
        parser = BSARecBase.parse_model_args(parser)
        # (重要) 确保我们能接收 --loss 参数
        parser.add_argument('--loss', type=str, default='BPR',help='Type of loss function.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)

        # --- “隐藏Bug”修复 ---
        # 在 _base_init 之前，把 SASRec 的参数名 "嫁接" 给 BSARec 的参数名
        args.hidden_size = args.emb_size
        args.hidden_dropout_prob = self.dropout # self.dropout 来自 SequentialModel
        # --- 修复结束 ---

        self._base_init(args, corpus)

        # 告诉 loss 函数要用哪种损失
        self.loss_type = args.loss

    #
    # ↓↓↓ 步骤 1：修改 BSARecBase 的 forward (在 BSARecBase 类中) ↓↓↓
    #
    # (找到 BSARecBase，用这个 forward 替换它的 forward)
    #
    # def forward(self, feed_dict):
    #     ... (forward 开头的所有代码不变) ...
    #
    #     i_vectors = self.i_embeddings(i_ids)
    #     prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
    #
    #     u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
    #     i_v = i_vectors
    #
    #     # ↓↓↓ 修改这一行 ↓↓↓
    #     # 我们多返回一个 'his_vector'，给 CE Loss 使用
    #     return {'prediction': prediction.view(batch_size, -1), 'u_v': u_v, 'i_v': i_v,
    #             'his_vector': his_vector}

    #
    # ↓↓↓ 步骤 2：重写 BSARec (主类) 的 forward 和 loss ↓↓↓
    #

    def forward(self, feed_dict):
        """
        修正后的 forward
        """
        out_dict = BSARecBase.forward(self, feed_dict)

        if self.loss_type.upper() == 'CE':
            # --- 修正点 ---
            # 必须在这里完成全量物品的打分计算
            # 这样 BaseRunner 拿到的就是 [Batch, Item_Num] 的矩阵，而不是 [Batch, Emb_Size]
            # 也就不会报 IndexError 了
            
            # 1. 获取用户向量 [B, D]
            user_emb = out_dict['his_vector']
            # 2. 获取所有物品向量 [N, D]
            all_items = self.i_embeddings.weight
            # 3. 计算分数 [B, N]
            logits = torch.matmul(user_emb, all_items.transpose(0, 1))
            
            return {'prediction': logits} 
        else:
            return {'prediction': out_dict['prediction']}

    def loss(self, out_dict, feed_dict):
        """
        修正后的 loss
        """
        if self.loss_type.upper() != 'CE':
            return super().loss(out_dict)

        # --- CE Loss 的实现 ---
        
        # 因为 forward 已经计算好了 logits，这里直接拿来用
        logits = out_dict['prediction'] # [batch_size, item_num]

        # 获取目标物品 (Answer)
        answers = feed_dict['item_id'][:, 0]
        answers = answers.long()

        # 计算 Cross Entropy
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, answers)

        return loss


##以下代码直接复制论文开源代码
class LayerNorm(nn.Module):
    "从 BSARec 源码复制的 TF-style LayerNorm"
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        # 修复：FFN 中间层维度 (d_ff) 扩大4倍
        self.feed_forward = PositionwiseFeedForward(d_model=args.emb_size,
                                                    d_ff= 4 * args.emb_size,
                                                     dropout=args.hidden_dropout_prob)

        # 修复：添加 Add & Norm 2
        self.LayerNorm2 = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        # 修复：实现标准的 "Add & Norm" 流程
        # 1. 注意力层
        layer_output = self.layer(hidden_states, attention_mask)
        # 3. FFN 层
        feedforward_output = self.feed_forward(layer_output)
        # 4. Add & Norm
        hidden_states = self.LayerNorm2(layer_output + self.dropout2(feedforward_output))
        return hidden_states

class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = layers.MY_MultiHeadAttention_Full(d_model=args.emb_size,
                                                  n_heads=args.num_heads)
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, input_tensor, input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states

class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states