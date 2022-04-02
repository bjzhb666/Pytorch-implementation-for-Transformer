"""
Created on 2022.04.01
Paper: Attention is all you need.
Implementation of additive attention and the NMT using transformer
Author: ZHB
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from d2l.torch import d2l
# 定义可视化注意力机制的函数

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(4, 4),
                  cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True,
                                 squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

def test_show_heatmaps():
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    d2l.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
    d2l.plt.show()

def masked_softmax(scores, valid_len):
    """
    通过在最后⼀个轴上掩蔽元素来执⾏softmax操作，要在键值对的纬度做softmax
    :arg: X:3D张量，valid_len:1D或2D张量
    :return: 3D vector，形状同X
    """
    if valid_len is None: # 在 Python 中，有一个特殊的常量 None（N 必须大写）。和 False 不同，它不表示 0，也不表示空字符串，而表示没有值，也就是空值。
        return torch.softmax(scores, dim=-1)
    else:
        shape = scores.shape # （batch_size，查询的个数，键值对的个数）
        if valid_len.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_len, shape[1]) # 让valid_lens变成查询数个valid_len，还是1D
            # https://blog.csdn.net/weixin_45261707/article/details/119187799
        else:
            valid_lens = valid_len.reshape(-1) # 变成1D
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0，不能把softmax之后的结果置为0，softmax必须保证结果之和为1
        X = d2l.sequence_mask(scores.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# 加性注意力层,不管是自定义层、自定义块、自定义模型，都是通过继承Module类完成的
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hidden, dropout=0., **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # https://docs.pythontab.com/interpy/args_kwargs/Usage_kwargs/ **kwargs的用法，其实就是让程序可扩展性更好，后面并没有用到
        self.W_k = nn.Linear(key_size, num_hidden, bias=False)
        self.W_q = nn.Linear(query_size, num_hidden, bias=False)
        self.W_v = nn.Linear(num_hidden, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, key, value, valid_length): # forward函数需要传入真正的实参
        keys, queries = self.W_k(key), self.W_q(queries) # 类调用的时候需要写self
        """
           :arg: 输入queries的形状（batch_size，查询的个数，query_size）
            输入key的形状（batch_size，键值对的个数，key_size）
            维度扩展后，queries的形状（batch_size，查询的个数，1，num_hidden）
            维度扩展后，key的形状（batch_size，1，“键－值”对的个数，num_hidden)，维度扩展的目的是广播
           :return: 输出的形状（batch_size，查询的个数，value_size）
        """

        keys = keys.unsqueeze(1)
        queries = queries.unsqueeze(2)
        features =torch.tanh(queries + keys)
        # features的形状（batch_size，查询的个数，键值对的个数，num_hidden）
        scores = self.W_v(features).squeeze(-1) # 把最后为1的维度不要了,形状为（batch_size，查询的个数，键值对的个数）
        self.attention_weights = masked_softmax(scores, valid_length)
        return torch.bmm(self.dropout(self.attention_weights), value)

def test_additive_attention():
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hidden=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))

    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
    d2l.plt.show()

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, valid_len = None):
        """
        :param: key的形状（batch_size，键值对的个数，dimension）
                value的形状（batch_size，键值对的个数，value_size）
                query的形状（batch_size，查询的个数，dimension）
                valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
        :return

        """
        d = query.shape[-1]
        # query = query.transpose(1,2) # query的形状变为（batch_size，dimension，查询的个数）
        key = key.transpose(1, 2) # 因为希望score的形状保持（batch_size，查询的个数，键值对的个数），所以用query*key，因此key要做转置
        scores = torch.bmm(query, key) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_len)
        return torch.bmm(self.dropout(self.attention_weights), value)

def test_dot_product_attention():
    query = torch.normal(0, 1, (2, 3, 2))
    value = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    key =  torch.ones((2, 10, 2))
    valid_len = torch.tensor([2, 6])
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(key, value, query,valid_len))
    # d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
    #                  xlabel='Keys', ylabel='Queries')
    d2l.plt.show()

test_dot_product_attention()