import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class OutLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=1, stride=1, padding=0):
        super(OutLayer, self).__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channel_in, channel_out),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(channel_out, 1),
        )

    def forward(self, ipt):
        o = self.main(ipt)
        return o

class Model(nn.Module):
    def __init__(self, n_layers=6, channels_interval=24):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.out_range = OutLayer(self.n_layers * self.channels_interval, 16)
        self.out_depth = OutLayer(self.n_layers * self.channels_interval, 16)

    def forward(self, input):
        # print(f"输入层维度: {input.shape}")
        tmp = []
        o = input

        # Down Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            # print(f"下采样层 {i+1} 卷积后维度: {o.shape}")
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]
            # print(f"下采样层 {i+1} 降采样后维度: {o.shape}")

        o = self.middle(o)
        # 输出距离和深度
        rr_pred = self.out_range(o)
        rd_pred = self.out_depth(o)

        # print(f"中间层维度: {o.shape}")

        return rr_pred, rd_pred


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, n_layers, channels_interval):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = Model(n_layers=n_layers, channels_interval=channels_interval)
        self.task_num = 2
        self.log_vars = nn.Parameter(torch.zeros(self.task_num))

    def forward(self, input_feature, target):
        outputs = self.model(input_feature)
        # 计算任务权重
        c_tau = torch.exp(self.log_vars)
        c_tau_squared = c_tau ** 2
        # 计算任务损失
        lossLocR = torch.sum((target[0] - outputs[0]) ** 2., -1)
        lossLocD = torch.sum((target[1] - outputs[1]) ** 2., -1)
        # 计算加权损失
        weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[0]
        weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[1]
        # 计算正则化项
        reg_loss = torch.log(1 + c_tau_squared).sum()
        # 计算总损失
        mtl_loss = weighted_lossLocR + weighted_lossLocD + reg_loss
        # 对批次内的损失进行平均，得到标量损失
        mtl_loss = mtl_loss.mean()

        return mtl_loss, self.log_vars.data.tolist(), outputs

if __name__ == '__main__':
    # 创建模型实例
    model = Model(n_layers=8, channels_interval=24)
    
    # 创建测试数据，维度为 torch.Size([1, 1600])
    # 假设数据形状为 [batch_size, sequence_length] = [1, 1600]
    # 但模型期望的是 [batch_size, channels, sequence_length] = [1, 1, 1600]
    batch_size = 3
    sequence_length = 1600
    test_input = torch.randn(batch_size, 1, sequence_length)
    
    print(f"输入数据维度: {test_input.shape}")
    
    # 将数据传入模型
    with torch.no_grad():  # 在测试阶段不需要计算梯度
        rr_pred, rd_pred = model(test_input)
    
    print(f"输出数据维度: {rr_pred.shape}, {rd_pred.shape}")
    
    # 验证损失模块
    loss_fn = MultiTaskLossWrapper(n_layers=8, channels_interval=24)
    
    # 创建测试目标数据
    rr_target = torch.randn(batch_size, 1)
    rd_target = torch.randn(batch_size, 1)
    
    # 计算损失
    total_loss, log_vars, outputs = loss_fn(test_input, (rr_target, rd_target))
    print(f"total_loss.shape: {total_loss.shape}")
    total_loss.backward()
    
    print(f"总损失: {total_loss.item():.4f}")
    print(f"距离: {outputs[0]}") 
    print(f"深度: {outputs[1]}")
