import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

# 导入自定义模块
from unet_basic import Model
from dataloader import HtDataset

class Trainer:
    """训练器类，用于训练UNet模型"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader = None, 
                 criterion: nn.Module = None, optimizer: optim.Optimizer = None, device: torch.device = None):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            criterion: 损失函数
            optimizer: 优化器
            device: 计算设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-3)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        返回:
            平均训练损失
        """
        self.model.train()  # 设置模型为训练模式
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # 获取输入和标签数据
            inputs = batch_data['ht_input'].to(self.device)
            targets = batch_data['ht_label'].to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}')
                
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> float:
        """
        验证模型
        
        返回:
            平均验证损失
        """
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():  # 禁用梯度计算
            for batch_data in self.val_loader:
                # 获取输入和标签数据
                inputs = batch_data['ht_input'].to(self.device)
                targets = batch_data['ht_label'].to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 累计损失
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints') -> None:
        """
        训练模型
        
        参数:
            num_epochs: 训练轮数
            save_dir: 模型保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练，设备: {self.device}")
        print(f"训练数据批次数量: {len(self.train_loader)}")
        if self.val_loader:
            print(f"验证数据批次数量: {len(self.val_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 30)
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f'训练损失: {train_loss:.6f}')
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            print(f'验证损失: {val_loss:.6f}')
            
            # # 保存最佳模型
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     model_path = os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth')
            #     torch.save({
            #         'epoch': epoch+1,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'train_loss': train_loss,
            #         'val_loss': val_loss,
            #     }, model_path)
            #     print(f'保存最佳模型: {model_path}')
            
            # # 每10个epoch保存一次检查点
            # if (epoch + 1) % 10 == 0:
            #     checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            #     torch.save({
            #         'epoch': epoch+1,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'train_losses': self.train_losses,
            #         'val_losses': self.val_losses,
            #     }, checkpoint_path)
            #     print(f'保存检查点: {checkpoint_path}')
        
        print('\n训练完成!')
        
    def plot_losses(self) -> None:
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失')
        if self.val_losses:
            plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练过程中的损失变化')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def save_losses(self, file_path: str) -> None:
        """保存损失数据到文件"""
        np.savez(file_path, train_losses=np.array(self.train_losses), 
                 val_losses=np.array(self.val_losses))
        print(f'损失数据已保存到: {file_path}')
        
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        测试模型
        
        参数:
            test_loader: 测试数据加载器
            
        返回:
            包含测试指标的字典
        """
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0.0
        num_batches = 0
        predictions = []
        targets_list = []
        
        with torch.no_grad():  # 禁用梯度计算
            for batch_data in test_loader:
                # 获取输入和标签数据
                inputs = batch_data['ht_input'].to(self.device)
                targets = batch_data['ht_label'].to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 累计损失
                total_loss += loss.item()
                num_batches += 1
                
                # 保存预测结果和真实标签用于后续分析
                predictions.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 合并所有批次的结果
        if predictions and targets_list:
            predictions = np.concatenate(predictions, axis=0)
            targets_list = np.concatenate(targets_list, axis=0)
            
            # 计算其他评估指标（如MAE）
            mae = np.mean(np.abs(predictions - targets_list))
            
            return {
                'test_loss': avg_loss,
                'test_mae': mae,
                'predictions': predictions,
                'targets': targets_list
            }
        else:
            return {
                'test_loss': avg_loss,
                'test_mae': 0.0,
                'predictions': np.array([]),
                'targets': np.array([])
            }
    
    def plot_test_results(self, test_results: Dict[str, Any]) -> None:
        """
        绘制测试结果
        
        参数:
            test_results: 测试结果字典
        """
        predictions = test_results['predictions']
        targets = test_results['targets']
        
        if len(predictions) == 0 or len(targets) == 0:
            print("没有测试结果可绘制")
            return
            
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('测试结果可视化', fontsize=16)
        
        # 选择前4个样本进行可视化
        num_samples = min(4, len(predictions))
        
        for i in range(num_samples):
            row = i // 2
            col = i % 2
            
            # 绘制预测结果和真实标签
            axes[row, col].plot(targets[i].flatten(), label='真实标签', alpha=0.7)
            axes[row, col].plot(predictions[i].flatten(), label='预测结果', alpha=0.7)
            axes[row, col].set_title(f'样本 {i+1}')
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 绘制散点图：预测值 vs 真实值
        plt.figure(figsize=(8, 6))
        plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')
        plt.grid(True)
        plt.show()
        
        # 打印测试指标
        print(f"测试损失: {test_results['test_loss']:.6f}")
        print(f"平均绝对误差 (MAE): {test_results['test_mae']:.6f}")


def main():
    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    """主函数，演示如何使用训练器"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据集和数据加载器
    # 注意：请根据实际情况修改数据路径
    data_dir = r'E:\4.0Dr\WPDP\dataset\ht_simple'  # 请根据实际情况修改
    
    # 读取数据划分文件
    with open('dataset_split.json', 'r') as f:
        dataset_split = json.load(f)
        train_files = dataset_split['train']
        val_files = dataset_split['val']
        test_files = dataset_split['test']
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录 {data_dir} 不存在，请修改为正确的路径")
        return
    
    # 创建数据集
    train_dataset = HtDataset(data_dir, train_files)
    val_dataset = HtDataset(data_dir, val_files)
    test_dataset = HtDataset(data_dir, test_files)
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    # 创建数据加载器
    batch_size = 4  # 根据内存情况调整
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}')
    print(f'训练批次数量: {len(train_loader)}, 验证批次数量: {len(val_loader)}, 测试批次数量: {len(test_loader)}')
    
    # 创建模型
    model = Model(n_layers=2, channels_interval=24)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')
    
    # 创建损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # 开始训练
    num_epochs = 300
    trainer.train(num_epochs=num_epochs, save_dir='./checkpoints')
    
    # 绘制损失曲线
    trainer.plot_losses()
    
    # 保存损失数据
    trainer.save_losses('./checkpoints/losses.npz')
    
    # 在测试集上进行测试
    print("\n=== 开始测试 ===")
    test_results = trainer.test(test_loader)
    print(f"测试完成:")
    print(f"  测试损失: {test_results['test_loss']:.6f}")
    print(f"  平均绝对误差 (MAE): {test_results['test_mae']:.6f}")
    
    # 绘制测试结果
    trainer.plot_test_results(test_results)


if __name__ == '__main__':
    main()