import torch
from d2l.torch import d2l
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    这里面num_hiddens就代表了参数po的维度（po即没有除以multihead的维度）这里真正单个注意力的hidden size是num_hiddens/num_heads
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        :param
        queries的形状：（batch_size，查询的个数，query_size）
        keys的形状：（batch_size，键值对的个数，key_size)
        values的形状：（batch_size，键值对的个数，values_size）
        valid_lens的形状（batch_size，）或者（batch_size，有效长度）

        :return:
        """
        queries = transpose_qkv(self.W_q(queries), self.heads)
        keys = transpose_qkv(self.W_k(keys), self.heads)
        values =transpose_qkv(self.W_v(values), self.heads)
        #  transpose_qkv后的输出是一个3D张量，正好可以输入进attention中

        if valid_lens is not None:
            # 在轴0即batch_size维度，将第⼀项（标量或者⽮量）复制num_heads次，然后如此复制第⼆项，然后诸如此类。
            # 这个多头注意力的实现其实是用投影后num_hidden的降低换来了多头，但是valid_lens没有被翻倍。所以这里在batch_size维度将其扩展
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.heads, dim=0)

        scores = self.attention(queries, keys, values, valid_lens)
        outputs = transpose_output(scores, self.heads)

        return self.W_o(outputs)


def transpose_qkv(X, num_heads):
    """
    为了并行计算，将输入X的最后一维切成 num_hidden/num_heads，然后再扩展一维作为num_heads，这样只做一次投影
    :param:
        X的维度：（batch_size，查询的个数/键值对的个数，num_hidden）
    :return:
        最终返回的形状:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3) # 变换后X的维度（batch_size，num_heads，键值对的个数/查询的个数，num_hidden/num_heads）
    # 最终返回的形状:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """
    transpose_qkv的逆操作
    :return :（batch_size，查询的个数/键值对的个数，num_hidden）
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1) # 相当于把num_heads和num_hidden/num_heads又乘回去了

def test_Multi_Head_attention():
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,num_hiddens, num_heads, 0.5)
    print(attention.eval())

    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)