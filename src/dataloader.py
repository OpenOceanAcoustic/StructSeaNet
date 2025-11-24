import numpy as np
import os
import matplotlib.pyplot as plt
import struct
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader

def read_ht_bin(datapth: str) -> Dict[str, Any]:
    """
    读取HT二进制文件
    
    参数:
        datapth: HT文件路径
    
    返回:
        Dict[str, Any]: 包含ht_label和ht_input等数据的字典
    """
    
    # 检查文件是否存在
    if not os.path.exists(datapth):
        raise FileNotFoundError(f"文件不存在: {datapth}")
    
    try:
        # 读取二进制文件
        with open(datapth, 'rb') as f:
            Ns = struct.unpack('<I', f.read(4))[0]  # '<' 表示小端字节序（MATLAB默认），'I' 对应 uint32
            # 读取 float32 类型数据（对应 MATLAB 的 'float32'）
            Sd = struct.unpack('<f', f.read(4))[0]  # '<' 小端字节序，'f' 对应 float32
            Rr = struct.unpack('<f', f.read(4))[0]
            Rd = struct.unpack('<f', f.read(4))[0]
            max_ps_log_sp = struct.unpack('<f', f.read(4))[0]
            min_ps_log_sp = struct.unpack('<f', f.read(4))[0]
            max_ph_log_sp = struct.unpack('<f', f.read(4))[0]
            min_ph_log_sp = struct.unpack('<f', f.read(4))[0]
            # 读取数据长度（假设数据为float32格式）
            data = np.fromfile(f, dtype=np.float32)
        
        # 假设ht_r和signal_cut_r各有Ns个点
        ht_label = data[0:Ns]
        ht_input = data[Ns:Ns*2]
        
        return {
            'Ns': Ns,
            'Sd': Sd,
            'Rr': Rr,
            'Rd': Rd,
            'max_ps_log_sp': max_ps_log_sp,
            'min_ps_log_sp': min_ps_log_sp,
            'max_ph_log_sp': max_ph_log_sp,
            'min_ph_log_sp': min_ph_log_sp,
            'ht_label': ht_label,
            'ht_input': ht_input
        }
    
    except Exception as e:
        raise RuntimeError(f"读取文件 {datapth} 时出错: {e}")
    

class HtDataset(Dataset):
    """HT数据集类"""
    
    def __init__(self, file_path_list: List[str] = None):
        """
        初始化数据集
        
        参数:
            file_path_list: 文件路径列表
        """
        self.file_paths = file_path_list
    
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
            'Rd': data_dict['Rd'],
            'max_ps_log_sp': data_dict['max_ps_log_sp'],
            'min_ps_log_sp': data_dict['min_ps_log_sp'],
            'max_ph_log_sp': data_dict['max_ph_log_sp'],
            'min_ph_log_sp': data_dict['min_ph_log_sp'],
        }


if __name__ == '__main__':
    # 原始测试代码
    htdname = 'HTD042'
    j = 1
    base_dir = r'..\datasets\ht_denoise_log_32Hz'
    datapth = os.path.join(base_dir, f"sig300-{htdname}-{j}.bin")
    print(datapth)
    data = read_ht_bin(datapth)
    
    # 使用新数据集类的示例
    # 创建文件路径列表
    file_paths = []
    if os.path.exists(base_dir):
        for file_name in os.listdir(base_dir):
            if file_name.endswith('.bin'):
                file_paths.append(os.path.join(base_dir, file_name))
    
    dataset = HtDataset(file_paths)
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