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

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

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

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

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
        # print(f"中间层维度: {o.shape}")

        # Up Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # print(f"上采样层 {i+1} 插值后维度: {o.shape}")
            # Skip Connection
            # 确保尺寸匹配
            tmp_out = tmp[self.n_layers - i - 1]
            if o.shape[2] != tmp_out.shape[2]:
                # 调整尺寸以匹配
                min_len = min(o.shape[2], tmp_out.shape[2])
                o = o[:, :, :min_len]
                tmp_out = tmp_out[:, :, :min_len]
            o = torch.cat([o, tmp_out], dim=1)
            # print(f"上采样层 {i+1} 跳跃连接后维度: {o.shape}")
            o = self.decoder[i](o)
            # print(f"上采样层 {i+1} 卷积后维度: {o.shape}")

        # Final layer - ensure size matching with input
        if o.shape[2] != input.shape[2]:
            min_len = min(o.shape[2], input.shape[2])
            o = o[:, :, :min_len]
            input = input[:, :, :min_len]
            
        o = torch.cat([o, input], dim=1)
        # print(f"最终层拼接后维度: {o.shape}")
        o = self.out(o)
        # print(f"输出层维度: {o.shape}")
        return o

if __name__ == '__main__':
    # 创建模型实例
    model = Model(n_layers=8, channels_interval=24)
    
    # 创建测试数据，维度为 torch.Size([1, 1600])
    # 假设数据形状为 [batch_size, sequence_length] = [1, 1600]
    # 但模型期望的是 [batch_size, channels, sequence_length] = [1, 1, 1600]
    batch_size = 1
    sequence_length = 1600
    test_input = torch.randn(batch_size, 1, sequence_length)
    
    print(f"输入数据维度: {test_input.shape}")
    
    # 将数据传入模型
    with torch.no_grad():  # 在测试阶段不需要计算梯度
        output = model(test_input)
    
    print(f"输出数据维度: {output.shape}")
    
    # 验证输入和输出形状是否一致
    print(f"输入和输出形状是否一致: {test_input.shape == output.shape}")
    