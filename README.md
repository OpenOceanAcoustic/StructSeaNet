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
2. 修改 `trainer.py` 中的数据路径
3. 运行训练器：
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

## 模型保存

训练过程中会自动保存：
- 最佳模型：验证损失最低的模型
- 检查点：每10个epoch保存一次

模型和损失数据保存在 `checkpoints/` 目录中。
