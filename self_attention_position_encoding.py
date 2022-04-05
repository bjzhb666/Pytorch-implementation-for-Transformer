import d2l.torch as d2l
import torch
import torch.nn as nn


# position encoding
class PositionEncoding(nn.Module):
    def __init__(self, num_hidden, dropout, max_length=1000,  **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        # 主要工作生成P，1是batch_size，后面和真正的batch_size做广播，思想是先产生一个足够大的P，然后在里面截取
        self.P = torch.zeros((1, max_length, num_hidden))
        x = torch.arange(max_length, dtype=torch.float32)\
                .reshape(-1, 1)/torch.pow(10000, torch.arange(0,num_hidden,2,dtype=torch.float32)/num_hidden)
        # 特别注意使用arange的时候要标准数据类型
        # torch.arange产生的是 torch.Size([max_length])，reshape以后变成二维的tensor了
        self.P[:,:,0::2] = torch.sin(x)
        self.P[:,:,1::2] = torch.cos(x)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1],:].to(X.device)
        return self.dropout(X)

def test_PositionEncoding():
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 3.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
    d2l.plt.show()

if __name__=='main':
    test_PositionEncoding()
