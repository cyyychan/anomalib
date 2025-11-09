import sys
import os
# 添加anomalib主目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from anomalib.data import SmoreAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# 这两个txt可以自己指定，假如训练集和测试集一直则设置成一样的即可
# 假如是图片文件数据可以通过parse_image.py生成txt文件
# Initialize components
datamodule = SmoreAD("/dataset/siyuanchen/AD/datasets/Anamalib_test/train.txt", "/dataset/siyuanchen/AD/datasets/Anamalib_test/test.txt")
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)

# Test the model
engine.test(datamodule=datamodule, model=model)
