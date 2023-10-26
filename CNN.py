import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm

from data_process import TimeSeriesDataset, data_process

train_data, _ = data_process()

# ########## 测试（不加时间数据）
ground_truth = np.array([train_data['active_index'], train_data['consume_index']])
ground_truth = list(zip(*ground_truth))
# ########## 测试

# 示例数据
data = np.array(train_data)

X = data[:, 2:-2].astype(float)
y = data[:, -2:].astype(float)

# # 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
#
# 创建数据加载器
batch_size = 30
dataset = TimeSeriesDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.Mish()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        """
        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)  # 最后一层全连接层

    def forward(self, x):
        """
        :param x: (batch_size, input_channel, seq_len)
        :return:
        """
        y1 = self.tcn(x)  # (batch_size, hidden_channel, seq_len)
        y1 = y1[:, :, -1]  # (batch_size, hidden_channel)
        return self.linear(y1)  # (batch_size, output_size)


# 参数设置
input_size = 1
output_size = 2
num_channels = [68] * 4
kernel_size = 2
dropout = 0.2
model = TCN(input_size, output_size, num_channels, kernel_size, dropout)

# 损失函数
criterion = nn.MSELoss()

# 优化器
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for i, (X, y) in enumerate(dataloader):
        y_pred = model(X.unsqueeze(1))
        loss = criterion(y_pred, y)
        loss = torch.sqrt(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 每20个batch打印一次损失
        if (i + 1) % 20 == 0:
            print('Epoch {}, Batch {}/{}, Loss {}'.format(epoch + 1, i + 1, len(dataloader), loss.item()))

    # 每个epoch打印一次平均损失
    print('Epoch {}, Loss {}'.format(epoch + 1, total_loss / len(dataloader)))
