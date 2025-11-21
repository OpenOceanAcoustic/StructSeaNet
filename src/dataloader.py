import numpy as np
import os
import matplotlib.pyplot as plt
import struct
from typing import Dict, Any

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
    



if __name__ == '__main__':
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
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(data['ht_label'], label='ht_label')
    plt.plot(data['ht_input'], label='ht_input')
    plt.legend()
    plt.title(f"{htdname}-{j}")
    plt.show()