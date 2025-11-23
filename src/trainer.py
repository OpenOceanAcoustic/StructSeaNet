import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import datetime
import yaml

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
        
        # 添加时间戳属性
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        # 创建基于时间戳的结果保存目录
        timestamp_save_dir = os.path.join('../results', self.timestamp)
        os.makedirs(timestamp_save_dir, exist_ok=True)
        # 同时创建checkpoints子目录
        checkpoints_dir = os.path.join(timestamp_save_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # 如果提供了save_dir参数，则也创建该目录（保持向后兼容）
        if save_dir != './checkpoints':
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
            
            # 保存最佳模型到基于时间戳的目录
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存到时间戳目录
                timestamp_save_dir = os.path.join('../results', self.timestamp)
                checkpoints_dir = os.path.join(timestamp_save_dir, 'checkpoints')
                model_path = os.path.join(checkpoints_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, model_path)
                print(f'保存最佳模型: {model_path}')
                
                # 如果提供了save_dir参数且不等于默认值，则也保存到该目录（保持向后兼容）
                if save_dir != './checkpoints':
                    model_path_compat = os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth')
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, model_path_compat)
                    print(f'保存最佳模型(兼容): {model_path_compat}')
            
            # 最后一个epoch保存检查点
            if (epoch + 1)  == num_epochs:
                # 保存到时间戳目录
                timestamp_save_dir = os.path.join('../results', self.timestamp)
                checkpoints_dir = os.path.join(timestamp_save_dir, 'checkpoints')
                checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, checkpoint_path)
                print(f'保存检查点: {checkpoint_path}')
                
                # 如果提供了save_dir参数且不等于默认值，则也保存到该目录（保持向后兼容）
                if save_dir != './checkpoints':
                    checkpoint_path_compat = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                    }, checkpoint_path_compat)
                    print(f'保存检查点(兼容): {checkpoint_path_compat}')
        
        print('\n训练完成!')
        
    def plot_losses(self, save_to_results: bool = True) -> None:
        """绘制训练和验证损失曲线
        
        参数:
            save_to_results: 是否将图像保存到results文件夹中
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失')
        if self.val_losses:
            plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练过程中的损失变化')
        plt.legend()
        plt.grid(True)
        
        # 如果需要保存到results文件夹
        if save_to_results:
            # 创建基于时间戳的结果保存目录
            timestamp_save_dir = os.path.join('../results', self.timestamp)
            os.makedirs(timestamp_save_dir, exist_ok=True)
            
            # 保存图像
            loss_plot_path = os.path.join(timestamp_save_dir, 'loss_curve.png')
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            print(f'损失曲线图已保存到: {loss_plot_path}')
        
        plt.show()
        
    def save_losses(self, file_path: str = None) -> None:
        """保存损失数据到文件
        
        参数:
            file_path: 保存文件路径，如果为None则保存到results文件夹中
        """
        # 如果没有提供文件路径，则保存到results文件夹中
        if file_path is None:
            # 创建基于时间戳的结果保存目录
            timestamp_save_dir = os.path.join('../results', self.timestamp)
            os.makedirs(timestamp_save_dir, exist_ok=True)
            
            # 设置默认文件路径
            file_path = os.path.join(timestamp_save_dir, 'losses.npz')
        
        np.savez(file_path, train_losses=np.array(self.train_losses), 
                 val_losses=np.array(self.val_losses))
        print(f'损失数据已保存到: {file_path}')
        
    def test(self, test_loader: DataLoader, save_to_results: bool = True) -> Dict[str, float]:
        """
        测试模型
        
        参数:
            test_loader: 测试数据加载器
            save_to_results: 是否将测试结果保存到results文件夹中
            
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
            
            # 如果需要保存到results文件夹
            if save_to_results:
                # 创建基于时间戳的结果保存目录
                timestamp_save_dir = os.path.join('../results', self.timestamp)
                os.makedirs(timestamp_save_dir, exist_ok=True)
                
                # 保存测试结果到JSON文件
                test_metrics = {
                    'test_loss': float(avg_loss),
                    'test_mae': float(mae)
                }
                
                test_metrics_path = os.path.join(timestamp_save_dir, 'test_metrics.json')
                with open(test_metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                print(f'测试指标已保存到: {test_metrics_path}')
                
                # 保存预测结果和真实标签
                test_results_path = os.path.join(timestamp_save_dir, 'test_results.npz')
                np.savez(test_results_path, predictions=predictions, targets=targets_list)
                print(f'测试结果已保存到: {test_results_path}')
            
            return {
                'test_loss': avg_loss,
                'test_mae': mae,
                'predictions': predictions,
                'targets': targets_list
            }
        else:
            # 如果需要保存到results文件夹
            if save_to_results:
                # 创建基于时间戳的结果保存目录
                timestamp_save_dir = os.path.join('../results', self.timestamp)
                os.makedirs(timestamp_save_dir, exist_ok=True)
                
                # 保存测试结果到JSON文件
                test_metrics = {
                    'test_loss': float(avg_loss),
                    'test_mae': 0.0
                }
                
                test_metrics_path = os.path.join(timestamp_save_dir, 'test_metrics.json')
                with open(test_metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                print(f'测试指标已保存到: {test_metrics_path}')
            
            return {
                'test_loss': avg_loss,
                'test_mae': 0.0,
                'predictions': np.array([]),
                'targets': np.array([])
            }
    
    def plot_test_results(self, test_results: Dict[str, Any], save_to_results: bool = True) -> None:
        """
        绘制测试结果
        
        参数:
            test_results: 测试结果字典
            save_to_results: 是否将测试结果图保存到results文件夹中，默认为True
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
        
        # 如果需要保存到results文件夹
        if save_to_results:
            # 创建基于时间戳的结果保存目录
            timestamp_save_dir = os.path.join('../results', self.timestamp)
            os.makedirs(timestamp_save_dir, exist_ok=True)
            
            # 保存测试结果图
            test_plot_path = os.path.join(timestamp_save_dir, 'test_results.png')
            fig.savefig(test_plot_path, dpi=300, bbox_inches='tight')
            print(f'测试结果图已保存到: {test_plot_path}')
        
        plt.show()
        
        # 绘制散点图：预测值 vs 真实值
        plt.figure(figsize=(8, 6))
        plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')
        plt.grid(True)
        
        # 如果需要保存到results文件夹
        if save_to_results:
            # 创建基于时间戳的结果保存目录
            timestamp_save_dir = os.path.join('../results', self.timestamp)
            os.makedirs(timestamp_save_dir, exist_ok=True)
            
            # 保存散点图
            scatter_plot_path = os.path.join(timestamp_save_dir, 'scatter_plot.png')
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
            print(f'散点图已保存到: {scatter_plot_path}')
        
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
    split_json = r'..\split\theTrueTrain.json'
    batch_size = 16  # 根据内存情况调整
    model_n_layers = 8
    model_channels_interval = 24
    optim_lr = 1e-3
    train_num_epochs = 300

    # 读取数据划分文件
    with open(split_json, 'r') as f:
        dataset_split = json.load(f)
        train_file_paths = dataset_split['train']
        val_file_paths = dataset_split['val']
        test_file_paths = dataset_split['test']
    
    
    # 创建数据集
    train_dataset = HtDataset(train_file_paths)
    val_dataset = HtDataset(val_file_paths)
    test_dataset = HtDataset(test_file_paths)
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}')
    print(f'训练批次数量: {len(train_loader)}, 验证批次数量: {len(val_loader)}, 测试批次数量: {len(test_loader)}')
    
    # 创建模型
    model = Model(n_layers=model_n_layers, channels_interval=model_channels_interval)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')
    
    # 创建损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=optim_lr)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # 保存超参数到YAML文件
    hyperparameters = {
        'split_json': split_json,
        'batch_size': batch_size,
        'model_n_layers': model_n_layers,
        'model_channels_interval': model_channels_interval,
        'optim_lr': optim_lr,
        'train_num_epochs': train_num_epochs,
        'device': str(device)
    }
    
    # 创建基于时间戳的结果保存目录
    timestamp_save_dir = os.path.join('../results', trainer.timestamp)
    os.makedirs(timestamp_save_dir, exist_ok=True)
    
    # 保存超参数到YAML文件
    hyperparams_path = os.path.join(timestamp_save_dir, 'train.yaml')
    with open(hyperparams_path, 'w', encoding='utf-8') as f:
        yaml.dump(hyperparameters, f, allow_unicode=True, default_flow_style=False)
    print(f'超参数已保存到: {hyperparams_path}')
    
    # 开始训练
    trainer.train(num_epochs=train_num_epochs, save_dir='./checkpoints')
    
    # 绘制损失曲线
    trainer.plot_losses(save_to_results=True)
    
    # 保存损失数据
    trainer.save_losses()
    
    # 在测试集上进行测试
    print("\n=== 开始测试 ===")
    test_results = trainer.test(test_loader, save_to_results=True)
    print(f"测试完成:")
    print(f"  测试损失: {test_results['test_loss']:.6f}")
    print(f"  平均绝对误差 (MAE): {test_results['test_mae']:.6f}")
    
    # 绘制测试结果
    trainer.plot_test_results(test_results, save_to_results=True)


if __name__ == '__main__':
    main()