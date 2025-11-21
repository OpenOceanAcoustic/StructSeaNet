import numpy as np
import os
import matplotlib.pyplot as plt
import struct
from typing import Dict, Any, List
import torch
from torch.utils import data

def read_ht_bin(datapth: str) -> Dict[str, Any]:
    """
    读取HT二进制文件
    
    参数:
        htdname: HT文件名
        j: 索引
        base_dir: 基础目录
    
    返回:
        Dict[str, Any]: 包含ht_r和signal_cut_r的字典
    """
    
    # 检查文件是否存在
    if not os.path.exists(datapth):
        raise FileNotFoundError(f"文件不存在: {datapth}")
    
    try:
        # 读取二进制文件
        with open(datapth, 'rb') as f:
            # 跳过前12字节
            Ns = struct.unpack('<I', f.read(4))[0]  # '<' 表示小端字节序（MATLAB默认），'I' 对应 uint32
    
            # 读取 float32 类型数据（对应 MATLAB 的 'float32'）
            Sd = struct.unpack('<f', f.read(4))[0]  # '<' 小端字节序，'f' 对应 float32
            Rr = struct.unpack('<f', f.read(4))[0]
            Rd = struct.unpack('<f', f.read(4))[0]
            # 读取数据长度（假设数据为float32格式）
            data = np.fromfile(f, dtype=np.float32)
        print(data.shape)
        
        # 假设ht_r和signal_cut_r各有160000个点
        ht_label = data[0:Ns:100]
        ht_input = data[Ns:Ns*2:100]
        
        return {
            'Ns': Ns,
            'Sd': Sd,
            'Rr': Rr,
            'Rd': Rd,
            'ht_label': ht_label,
            'ht_input': ht_input
        }
        
    except Exception as e:
        raise RuntimeError(f"读取文件 {datapth} 时出错: {e}")
    

class HtDataset(data.Dataset):
    """HT数据集类"""
    
    def __init__(self, data_dir: str, file_list: List[str] = None):
        """
        初始化数据集
        
        参数:
            data_dir: 数据文件所在的目录
            file_list: 文件列表，如果为None，则使用目录中的所有.bin文件
        """
        self.data_dir = data_dir
        if file_list is None:
            # 获取目录中所有的.bin文件
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.bin')]
        else:
            self.file_list = file_list
            
        # 确保文件路径完整
        self.file_paths = [os.path.join(self.data_dir, f) for f in self.file_list]
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            包含输入数据和标签的字典
        """
        # 读取数据
        file_path = self.file_paths[idx]
        data_dict = read_ht_bin(file_path)
        
        # 转换为torch张量
        ht_input = torch.tensor(data_dict['ht_input'], dtype=torch.float32)
        ht_label = torch.tensor(data_dict['ht_label'], dtype=torch.float32)
        
        # 确保数据维度正确
        if ht_input.dim() == 1:
            ht_input = ht_input.unsqueeze(0)  # 添加通道维度
        if ht_label.dim() == 1:
            ht_label = ht_label.unsqueeze(0)  # 添加通道维度
            
        return {
            'ht_input': ht_input,
            'ht_label': ht_label,
            'Ns': data_dict['Ns'],
            'Sd': data_dict['Sd'],
            'Rr': data_dict['Rr'],
            'Rd': data_dict['Rd']
        }


if __name__ == '__main__':
    # 原始测试代码
    htdname = 'HTD042'
    j = 1
    base_dir = r'E:\4.0Dr\WPDP\dataset\ht_denoise_log'
    datapth = os.path.join(base_dir, f"sig300-{htdname}-{j}.bin")
    data = read_ht_bin(datapth)
    print(data['ht_label'].shape)
    print(data['ht_input'].shape)
    print(data['Ns'])
    print(data['Sd'])
    print(data['Rr'])
    print(data['Rd'])
    
    # 使用新数据集类的示例
    dataset = HtDataset(base_dir)
    print(f"数据集大小: {len(dataset)}")
    
    # 获取第一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"输入数据形状: {sample['ht_input'].shape}")
        print(f"标签数据形状: {sample['ht_label'].shape}")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(data['ht_label'], label='ht_label')
    plt.plot(data['ht_input'], label='ht_input')
    plt.legend()
    plt.title(f"{htdname}-{j}")
    plt.show()