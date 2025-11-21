# StructSeaNet

一个神经网络学习海洋信道结构的仓库

## 项目结构

```
StructSeaNet/
├── src/
│   ├── dataloader.py     # 数据加载模块
│   ├── unet_basic.py     # UNet模型定义
│   └── trainer.py        # 训练器模块
└── README.md
```

## 模块说明

### 1. dataloader.py
包含数据加载和预处理功能：
- `read_ht_bin`: 读取HT二进制文件
- `HtDataset`: 自定义PyTorch数据集类

### 2. unet_basic.py
包含UNet模型定义：
- `DownSamplingLayer`: 下采样层
- `UpSamplingLayer`: 上采样层
- `Model`: 完整的UNet模型

### 3. trainer.py
包含训练器实现：
- `Trainer`: 训练器类，负责模型训练和验证

## 使用方法

1. 准备数据集，确保数据格式符合要求
2. 准备数据划分文件 `dataset_split.json`，指定训练集和验证集的文件列表
3. 修改 `trainer.py` 中的数据路径
4. 运行训练器：
   ```bash
   cd src
   python trainer.py
   ```

## 训练配置

可以在 `trainer.py` 的 `main()` 函数中调整以下参数：
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `n_layers`: UNet层数
- `channels_interval`: 通道间隔

## 数据划分文件

数据划分文件 `dataset_split.json` 的格式如下：

```json
{
  "train": [
    "E:\\4.0Dr\\WPDP\\dataset\\ht_simple\\sig300-HTD042-2"
  ],
  "test": [
    "E:\\4.0Dr\\WPDP\\dataset\\ht_simple\\sig300-HTD042-2"
  ]
}
```

注意：`test` 集合将被用作验证集。

请根据实际数据路径修改此文件。

## 测试和可视化

训练完成后，程序会自动在验证集上进行测试，并生成以下可视化结果：
- 训练和验证损失曲线
- 测试结果对比图（前4个样本的预测值与真实值对比）
- 预测值与真实值的散点图

## 模型保存

训练过程中会自动保存：
- 最佳模型：验证损失最低的模型
- 检查点：每10个epoch保存一次

模型和损失数据保存在 `checkpoints/` 目录中。
