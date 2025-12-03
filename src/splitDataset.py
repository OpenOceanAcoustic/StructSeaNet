import os
import json
import random

# 设置随机种子，确保划分结果可复现
random.seed(42)

# 定义文件夹路径和输出文件路径
folder_path = r"..\datasets\ht_denoise_log_32Hz"
output_file = "../split/theTrueTrain.json"

# 获取文件夹中所有.bin文件的路径
bin_files = []
for filename in os.listdir(folder_path):
    if filename.endswith(".bin"):
        # 拼接完整路径
        file_path = os.path.join(folder_path, filename)
        # 转换为Windows风格的反斜杠路径
        bin_files.append(file_path.replace('/', '\\'))

# 检查是否有bin文件
if not bin_files:
    raise ValueError("指定路径下没有找到任何.bin文件")

# 打乱文件顺序
random.shuffle(bin_files)

# 计算划分比例（7:1:2）
total = len(bin_files)
train_size = int(total * 0.7)
val_size = int(total * 0.1)
# 测试集大小为剩余部分（确保总和正确）
test_size = int(total * 0.2) #total - train_size - val_size

# 划分数据集
train_files = bin_files[:train_size]
val_files = bin_files[train_size:train_size + val_size]
# test_files = bin_files[train_size + val_size:]
test_files = bin_files[train_size + val_size:train_size + val_size + test_size]
# val_files = bin_files[:val_size]
# test_files = bin_files[:test_size]

# 构建JSON数据结构
data = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

# 写入JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"数据集划分完成，已保存到{output_file}")
print(f"总文件数: {total}, 训练集: {len(train_files)}, 验证集: {len(val_files)}, 测试集: {len(test_files)}")