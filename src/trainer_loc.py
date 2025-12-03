import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import scipy.io
import json
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import datetime
import re
import yaml
import argparse

# 导入自定义模块
from unet_loc import MultiTaskLossWrapper
from dataloader import HtDataset

Rrmax = 105
Rdmax = 5002

class Trainer:
    """训练器类，用于训练定位网络模型"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader = None, 
                 optimizer: optim.Optimizer = None, device: torch.device = None):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            optimizer: 优化器
            device: 计算设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-3)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_range_losses = []
        self.val_range_losses = []
        self.train_depth_losses = []
        self.val_depth_losses = []
        self.train_log_vars = []
        self.val_log_vars = []
        
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
        total_range_loss = 0.0
        total_depth_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # 获取输入和标签数据
            inputs = batch_data['ht_input'].to(self.device)
            rr_target = batch_data['Rr'].unsqueeze(1).to(self.device, dtype=torch.float32)
            rd_target = batch_data['Rd'].unsqueeze(1).to(self.device, dtype=torch.float32)
            # 对targets进行归一化
            rr_target = rr_target / Rrmax
            rd_target = rd_target / Rdmax
            targets = (rr_target, rd_target)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播和损失计算
            loss, log_vars, outputs = self.model(inputs, targets)
            rr_pred, rd_pred = outputs
            
            # 计算单独的任务损失（用于监控）
            range_loss = torch.mean((rr_target - rr_pred) ** 2).item()
            depth_loss = torch.mean((rd_target - rd_pred) ** 2).item()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_range_loss += range_loss
            total_depth_loss += depth_loss
            num_batches += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Total Loss: {loss.item():.6f}, Range Loss: {range_loss:.6f}, Depth Loss: {depth_loss:.6f}')
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_range_loss = total_range_loss / num_batches if num_batches > 0 else 0.0
        avg_depth_loss = total_depth_loss / num_batches if num_batches > 0 else 0.0
        
        # 记录训练历史
        self.train_losses.append(avg_loss)
        self.train_range_losses.append(avg_range_loss)
        self.train_depth_losses.append(avg_depth_loss)
        
        return avg_loss
    
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
        total_range_loss = 0.0
        total_depth_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():  # 禁用梯度计算
            for batch_data in self.val_loader:
                # 获取输入和标签数据
                inputs = batch_data['ht_input'].to(self.device)
                rr_target = batch_data['Rr'].unsqueeze(1).to(self.device, dtype=torch.float32)
                rd_target = batch_data['Rd'].unsqueeze(1).to(self.device, dtype=torch.float32)
                # 对targets进行归一化
                rr_target = rr_target / Rrmax
                rd_target = rd_target / Rdmax
                targets = (rr_target, rd_target)
                
                # 前向传播和损失计算
                loss, log_vars, outputs = self.model(inputs, targets)
                rr_pred, rd_pred = outputs
                
                # 计算单独的任务损失（用于监控）
                range_loss = torch.mean((rr_target - rr_pred) ** 2).item()
                depth_loss = torch.mean((rd_target - rd_pred) ** 2).item()
                
                # 累计损失
                total_loss += loss.item()
                total_range_loss += range_loss
                total_depth_loss += depth_loss
                num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_range_loss = total_range_loss / num_batches if num_batches > 0 else 0.0
        avg_depth_loss = total_depth_loss / num_batches if num_batches > 0 else 0.0
        
        # 记录验证历史
        self.val_losses.append(avg_loss)
        self.val_range_losses.append(avg_range_loss)
        self.val_depth_losses.append(avg_depth_loss)
        
        return avg_loss
    
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
            print(f'训练损失: {train_loss:.6f}')
            print(f'  距离损失: {self.train_range_losses[-1]:.6f}')
            print(f'  深度损失: {self.train_depth_losses[-1]:.6f}')
            
            # 验证
            val_loss = self.validate()
            print(f'验证损失: {val_loss:.6f}')
            print(f'  距离损失: {self.val_range_losses[-1]:.6f}')
            print(f'  深度损失: {self.val_depth_losses[-1]:.6f}')
            
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
                    'train_range_loss': self.train_range_losses[-1],
                    'val_range_loss': self.val_range_losses[-1],
                    'train_depth_loss': self.train_depth_losses[-1],
                    'val_depth_loss': self.val_depth_losses[-1],
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
                        'train_range_loss': self.train_range_losses[-1],
                        'val_range_loss': self.val_range_losses[-1],
                        'train_depth_loss': self.train_depth_losses[-1],
                        'val_depth_loss': self.val_depth_losses[-1],
                    }, model_path_compat)
                    print(f'保存最佳模型(兼容): {model_path_compat}')
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
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
                    'train_range_losses': self.train_range_losses,
                    'val_range_losses': self.val_range_losses,
                    'train_depth_losses': self.train_depth_losses,
                    'val_depth_losses': self.val_depth_losses,
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
                        'train_range_losses': self.train_range_losses,
                        'val_range_losses': self.val_range_losses,
                        'train_depth_losses': self.train_depth_losses,
                        'val_depth_losses': self.val_depth_losses,
                    }, checkpoint_path_compat)
                    print(f'保存检查点(兼容): {checkpoint_path_compat}')
        
        print('\n训练完成!')
        
    def plot_losses(self, save_to_results: bool = True) -> None:
        """
        绘制训练和验证损失曲线
        
        参数:
            save_to_results: 是否将图像保存到results文件夹中
        """
        # 创建3个子图，分别绘制总损失、距离损失和深度损失
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # 绘制总损失曲线
        axes[0].plot(self.train_losses, label='训练总损失', color='blue')
        if self.val_losses:
            axes[0].plot(self.val_losses, label='验证总损失', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失值')
        axes[0].set_title('训练过程中的总损失变化')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制距离损失曲线
        axes[1].plot(self.train_range_losses, label='训练距离损失', color='blue')
        if self.val_range_losses:
            axes[1].plot(self.val_range_losses, label='验证距离损失', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('损失值')
        axes[1].set_title('训练过程中的距离损失变化')
        axes[1].legend()
        axes[1].grid(True)
        
        # 绘制深度损失曲线
        axes[2].plot(self.train_depth_losses, label='训练深度损失', color='blue')
        if self.val_depth_losses:
            axes[2].plot(self.val_depth_losses, label='验证深度损失', color='red')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('损失值')
        axes[2].set_title('训练过程中的深度损失变化')
        axes[2].legend()
        axes[2].grid(True)
        
        # 调整子图间距
        plt.tight_layout()
        
        # 如果需要保存到results文件夹
        if save_to_results:
            # 创建基于时间戳的结果保存目录
            timestamp_save_dir = os.path.join('../results', self.timestamp)
            os.makedirs(timestamp_save_dir, exist_ok=True)
            
            # 保存图像
            loss_plot_path = os.path.join(timestamp_save_dir, 'loss_curves.png')
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            print(f'损失曲线图已保存到: {loss_plot_path}')
        
        plt.show()
        
    def save_losses(self, file_path: str = None) -> None:
        """
        保存损失数据到文件
        
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
        
        np.savez(file_path, 
                 train_losses=np.array(self.train_losses), 
                 val_losses=np.array(self.val_losses),
                 train_range_losses=np.array(self.train_range_losses),
                 val_range_losses=np.array(self.val_range_losses),
                 train_depth_losses=np.array(self.train_depth_losses),
                 val_depth_losses=np.array(self.val_depth_losses))
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
        
        # 存储预测结果和真实值
        rr_predictions = []
        rd_predictions = []
        rr_targets_list = []
        rd_targets_list = []
        
        # 存储其他数据
        Sd_list = np.array([])
        Rr_list = np.array([])
        Rd_list = np.array([])
        max_ps_log_sp_list = np.array([])
        min_ps_log_sp_list = np.array([])
        max_ph_log_sp_list = np.array([])
        min_ph_log_sp_list = np.array([])
        
        with torch.no_grad():  # 禁用梯度计算
            for batch_data in test_loader:
                # 获取输入和标签数据
                inputs = batch_data['ht_label'].to(self.device)
                rr_target = batch_data['Rr'].unsqueeze(1).to(self.device, dtype=torch.float32)
                rd_target = batch_data['Rd'].unsqueeze(1).to(self.device, dtype=torch.float32)
                # 对targets进行归一化
                rr_target = rr_target / Rrmax
                rd_target = rd_target / Rdmax
                targets = (rr_target, rd_target)
                
                # 获取其他数据
                Sd = batch_data['Sd']
                Rr = batch_data['Rr']
                Rd = batch_data['Rd']
                max_ps_log_sp = batch_data['max_ps_log_sp']
                min_ps_log_sp = batch_data['min_ps_log_sp']
                max_ph_log_sp = batch_data['max_ph_log_sp']
                min_ph_log_sp = batch_data['min_ph_log_sp']
                
                # 前向传播和损失计算
                loss, log_vars, outputs = self.model(inputs, targets)
                rr_pred, rd_pred = outputs
                
                # 累计损失
                total_loss += loss.item()
                num_batches += 1
                
                # 保存预测结果和真实标签用于后续分析
                rr_predictions.append(rr_pred.cpu().numpy())
                rd_predictions.append(rd_pred.cpu().numpy())
                rr_targets_list.append(rr_target.cpu().numpy())
                rd_targets_list.append(rd_target.cpu().numpy())
                
                # 保存其他数据
                Sd_list = np.append(Sd_list, Sd)
                Rr_list = np.append(Rr_list, Rr)
                Rd_list = np.append(Rd_list, Rd)
                max_ps_log_sp_list = np.append(max_ps_log_sp_list, max_ps_log_sp)
                min_ps_log_sp_list = np.append(min_ps_log_sp_list, min_ps_log_sp)
                max_ph_log_sp_list = np.append(max_ph_log_sp_list, max_ph_log_sp)
                min_ph_log_sp_list = np.append(min_ph_log_sp_list, min_ph_log_sp)
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 合并所有批次的结果
        if rr_predictions and rr_targets_list and rd_predictions and rd_targets_list:
            # 合并预测结果和真实值
            rr_predictions = np.concatenate(rr_predictions, axis=0)
            rd_predictions = np.concatenate(rd_predictions, axis=0)
            rr_targets_list = np.concatenate(rr_targets_list, axis=0)
            rd_targets_list = np.concatenate(rd_targets_list, axis=0)
            
            # 计算距离定位的评估指标
            rr_mse = np.mean((rr_targets_list - rr_predictions) ** 2)
            rr_mae = np.mean(np.abs(rr_targets_list - rr_predictions))
            rr_rmse = np.sqrt(rr_mse)
            
            # 计算深度定位的评估指标
            rd_mse = np.mean((rd_targets_list - rd_predictions) ** 2)
            rd_mae = np.mean(np.abs(rd_targets_list - rd_predictions))
            rd_rmse = np.sqrt(rd_mse)
            
            # 计算相对误差
            rr_relative_error = np.mean(np.abs(rr_targets_list - rr_predictions) / (np.abs(rr_targets_list) + 1e-8)) * 100
            rd_relative_error = np.mean(np.abs(rd_targets_list - rd_predictions) / (np.abs(rd_targets_list) + 1e-8)) * 100
            
            # 如果需要保存到results文件夹
            if save_to_results:
                # 创建基于时间戳的结果保存目录
                timestamp_save_dir = os.path.join('../results', self.timestamp)
                os.makedirs(timestamp_save_dir, exist_ok=True)
                
                # 保存测试结果到JSON文件
                test_metrics = {
                    'test_loss': float(avg_loss),
                    'range_metrics': {
                        'mse': float(rr_mse),
                        'mae': float(rr_mae),
                        'rmse': float(rr_rmse),
                        'relative_error': float(rr_relative_error)
                    },
                    'depth_metrics': {
                        'mse': float(rd_mse),
                        'mae': float(rd_mae),
                        'rmse': float(rd_rmse),
                        'relative_error': float(rd_relative_error)
                    }
                }
                
                test_metrics_path = os.path.join(timestamp_save_dir, 'test_metrics.json')
                with open(test_metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                print(f'测试指标已保存到: {test_metrics_path}')
                
                # 保存预测结果和真实标签
                test_results_path = os.path.join(timestamp_save_dir, 'test_results.npz')
                np.savez(test_results_path, 
                         rr_predictions=rr_predictions, 
                         rd_predictions=rd_predictions,
                         rr_targets=rr_targets_list,
                         rd_targets=rd_targets_list,
                         Sd=Sd_list, Rr=Rr_list, Rd=Rd_list,
                         max_ps_log_sp=max_ps_log_sp_list, min_ps_log_sp=min_ps_log_sp_list,
                         max_ph_log_sp=max_ph_log_sp_list, min_ph_log_sp=min_ph_log_sp_list)
                print(f'测试结果已保存到: {test_results_path}')
                
                # 转为.mat文件
                mat_file_path = os.path.join(timestamp_save_dir, 'test_results.mat')
                scipy.io.savemat(mat_file_path, {
                    'rr_predictions': rr_predictions, 
                    'rd_predictions': rd_predictions,
                    'rr_targets': rr_targets_list,
                    'rd_targets': rd_targets_list,
                    'Sd': Sd_list, 'Rr': Rr_list, 'Rd': Rd_list,
                    'max_ps_log_sp': max_ps_log_sp_list, 'min_ps_log_sp': min_ps_log_sp_list,
                    'max_ph_log_sp': max_ph_log_sp_list, 'min_ph_log_sp': min_ph_log_sp_list
                })
                print(f'测试结果已保存到: {mat_file_path}')
            
            return {
                'test_loss': avg_loss,
                'range_metrics': {
                    'mse': rr_mse,
                    'mae': rr_mae,
                    'rmse': rr_rmse,
                    'relative_error': rr_relative_error
                },
                'depth_metrics': {
                    'mse': rd_mse,
                    'mae': rd_mae,
                    'rmse': rd_rmse,
                    'relative_error': rd_relative_error
                },
                'rr_predictions': rr_predictions,
                'rd_predictions': rd_predictions,
                'rr_targets': rr_targets_list,
                'rd_targets': rd_targets_list
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
                    'range_metrics': {
                        'mse': 0.0,
                        'mae': 0.0,
                        'rmse': 0.0,
                        'relative_error': 0.0
                    },
                    'depth_metrics': {
                        'mse': 0.0,
                        'mae': 0.0,
                        'rmse': 0.0,
                        'relative_error': 0.0
                    }
                }
                
                test_metrics_path = os.path.join(timestamp_save_dir, 'test_metrics.json')
                with open(test_metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                print(f'测试指标已保存到: {test_metrics_path}')
            
            return {
                'test_loss': avg_loss,
                'range_metrics': {
                    'mse': 0.0,
                    'mae': 0.0,
                    'rmse': 0.0,
                    'relative_error': 0.0
                },
                'depth_metrics': {
                    'mse': 0.0,
                    'mae': 0.0,
                    'rmse': 0.0,
                    'relative_error': 0.0
                },
                'rr_predictions': np.array([]),
                'rd_predictions': np.array([]),
                'rr_targets': np.array([]),
                'rd_targets': np.array([])
            }
    
    def plot_test_results(self, test_results: Dict[str, Any], save_to_results: bool = True) -> None:
        """
        绘制测试结果
        
        参数:
            test_results: 测试结果字典
            save_to_results: 是否将测试结果图保存到results文件夹中，默认为True
        """
        # 获取预测结果和真实值
        rr_predictions = test_results['rr_predictions']
        rd_predictions = test_results['rd_predictions']
        rr_targets = test_results['rr_targets']
        rd_targets = test_results['rd_targets']
        
        if len(rr_predictions) == 0 or len(rr_targets) == 0 or len(rd_predictions) == 0 or len(rd_targets) == 0:
            print("没有测试结果可绘制")
            return
            
        # 创建图形，包含多个子图
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 距离预测 vs 真实值散点图
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(rr_targets, rr_predictions, alpha=0.5, color='blue')
        ax1.plot([rr_targets.min(), rr_targets.max()], [rr_targets.min(), rr_targets.max()], 'r--', lw=2)
        ax1.set_xlabel('真实距离值')
        ax1.set_ylabel('预测距离值')
        ax1.set_title('距离预测 vs 真实值')
        ax1.grid(True)
        
        # 2. 深度预测 vs 真实值散点图
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.scatter(rd_targets, rd_predictions, alpha=0.5, color='green')
        ax2.plot([rd_targets.min(), rd_targets.max()], [rd_targets.min(), rd_targets.max()], 'r--', lw=2)
        ax2.set_xlabel('真实深度值')
        ax2.set_ylabel('预测深度值')
        ax2.set_title('深度预测 vs 真实值')
        ax2.grid(True)
        
        # 3. 距离误差分布直方图
        ax3 = fig.add_subplot(2, 3, 3)
        rr_errors = rr_predictions - rr_targets
        ax3.hist(rr_errors, bins=50, alpha=0.7, color='blue')
        ax3.set_xlabel('距离预测误差')
        ax3.set_ylabel('频率')
        ax3.set_title('距离预测误差分布')
        ax3.grid(True)
        
        # 4. 深度误差分布直方图
        ax4 = fig.add_subplot(2, 3, 4)
        rd_errors = rd_predictions - rd_targets
        ax4.hist(rd_errors, bins=50, alpha=0.7, color='green')
        ax4.set_xlabel('深度预测误差')
        ax4.set_ylabel('频率')
        ax4.set_title('深度预测误差分布')
        ax4.grid(True)
        
        # 5. 距离误差随真实值变化图
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.scatter(rr_targets, rr_errors, alpha=0.5, color='blue')
        ax5.axhline(y=0, color='r', linestyle='--')
        ax5.set_xlabel('真实距离值')
        ax5.set_ylabel('距离预测误差')
        ax5.set_title('距离误差随真实值变化')
        ax5.grid(True)
        
        # 6. 深度误差随真实值变化图
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.scatter(rd_targets, rd_errors, alpha=0.5, color='green')
        ax6.axhline(y=0, color='r', linestyle='--')
        ax6.set_xlabel('真实深度值')
        ax6.set_ylabel('深度预测误差')
        ax6.set_title('深度误差随真实值变化')
        ax6.grid(True)
        
        plt.tight_layout()
        
        # 如果需要保存到results文件夹
        if save_to_results:
            # 创建基于时间戳的结果保存目录
            timestamp_save_dir = os.path.join('../results', self.timestamp)
            os.makedirs(timestamp_save_dir, exist_ok=True)
            
            # 保存测试结果图
            test_plot_path = os.path.join(timestamp_save_dir, 'test_results_visualization.png')
            fig.savefig(test_plot_path, dpi=300, bbox_inches='tight')
            print(f'测试结果可视化图已保存到: {test_plot_path}')
        
        plt.show()
        
        # 打印测试指标
        print("\n=== 测试指标 ===")
        print(f"测试总损失: {test_results['test_loss']:.6f}")
        print("\n距离定位指标:")
        print(f"  均方误差 (MSE): {test_results['range_metrics']['mse']:.6f}")
        print(f"  平均绝对误差 (MAE): {test_results['range_metrics']['mae']:.6f}")
        print(f"  均方根误差 (RMSE): {test_results['range_metrics']['rmse']:.6f}")
        print(f"  相对误差: {test_results['range_metrics']['relative_error']:.2f}%")
        print("\n深度定位指标:")
        print(f"  均方误差 (MSE): {test_results['depth_metrics']['mse']:.6f}")
        print(f"  平均绝对误差 (MAE): {test_results['depth_metrics']['mae']:.6f}")
        print(f"  均方根误差 (RMSE): {test_results['depth_metrics']['rmse']:.6f}")
        print(f"  相对误差: {test_results['depth_metrics']['relative_error']:.2f}%")


def main():
    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='UNet模型训练和测试')
    
    # 添加超参数命令行参数
    parser.add_argument('--test_mode', type=bool, default=False, help='是否为测试模式')
    parser.add_argument('--model_checkpoint_path', type=str, default=r'../results/20251122_130323/checkpoints/checkpoint_epoch_300.pth', help='预训练模型路径')
    parser.add_argument('--split_json', type=str, default=r'..\split\theTrueTrain.json', help='数据划分文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--model_n_layers', type=int, default=8, help='UNet模型的层数')
    parser.add_argument('--model_channels_interval', type=int, default=24, help='模型通道间隔')
    parser.add_argument('--optim_lr', type=float, default=1e-3, help='优化器学习率')
    parser.add_argument('--train_num_epochs', type=int, default=300, help='训练轮数')
    
    # 解析命令行参数
    args = parser.parse_args()

    """主函数，演示如何使用训练器"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据集和数据加载器
    # 注意：请根据实际情况修改数据路径
    test_mode = args.test_mode
    model_checkpoint_path = args.model_checkpoint_path
    split_json = args.split_json
    batch_size = args.batch_size  # 根据内存情况调整
    model_n_layers = args.model_n_layers
    model_channels_interval = args.model_channels_interval
    optim_lr = args.optim_lr
    train_num_epochs = args.train_num_epochs

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
    
    # 创建损失函数（包含模型）
    model = MultiTaskLossWrapper(n_layers=model_n_layers, channels_interval=model_channels_interval)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=optim_lr)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device
    )
    
    if test_mode:
        # 测试模式：加载预训练模型权重进行测试
        print("=== 测试模式 ===")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {model_checkpoint_path}")
        
        # 加载模型权重
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型权重: {model_checkpoint_path}")
        
        # 提取时间戳用于创建结果保存目录
        # 从路径中提取时间戳，例如从 '../results/20251122_130323/checkpoint_epoch_300.pth' 提取 '20251122_130323'
        timestamp_match = re.search(r'/results/(\d+_\d+)/', model_checkpoint_path)
        if timestamp_match:
            trainer.timestamp = timestamp_match.group(1)
        else:
            # 如果无法从路径提取时间戳，则使用当前时间戳
            trainer.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"使用时间戳: {trainer.timestamp}")
        
        # 在测试集上进行测试
        print("\n=== 开始测试 ===")
        test_results = trainer.test(test_loader, save_to_results=True)
        print(f"测试完成:")
        print(f"  测试损失: {test_results['test_loss']:.6f}")
        print(f"  距离均方误差 (MSE): {test_results['range_metrics']['mse']:.6f}")
        print(f"  深度均方误差 (MSE): {test_results['depth_metrics']['mse']:.6f}")
        
        # 绘制测试结果
        trainer.plot_test_results(test_results, save_to_results=True)
    else:
        # 训练模式
        print("=== 训练模式 ===")
        
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
        print(f"  距离均方误差 (MSE): {test_results['range_metrics']['mse']:.6f}")
        print(f"  深度均方误差 (MSE): {test_results['depth_metrics']['mse']:.6f}")
        
        # 绘制测试结果
        trainer.plot_test_results(test_results, save_to_results=True)


if __name__ == '__main__':
    main()