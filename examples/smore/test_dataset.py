import sys
import os
# 添加anomalib主目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from anomalib.data import SmoreAD
from anomalib.data import MVTecAD

# Initialize components
# datamodule = MVTecAD("/mnt/d/workspace/anomal-detect", category="bottle")
datamodule = SmoreAD("/mnt/d/workspace/anomal-detect/bottle/train.txt", "/mnt/d/workspace/anomal-detect/bottle/test.txt")
datamodule.setup()

# 获取第一个batch的数据
train_dataloader = datamodule.train_dataloader()
first_batch = next(iter(train_dataloader))

print("第一个batch的数据:")
for i, data in enumerate(first_batch):
    print(f"Batch {i}: {data}")
    if i >= 2:  # 只打印前3个样本
        break
