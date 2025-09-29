import sys
import os
# 添加anomalib主目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from anomalib.data import SmoreAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize components
datamodule = SmoreAD("/dataset/siyuanchen/AD/datasets/Anamalib_test/train.txt", "/dataset/siyuanchen/AD/datasets/Anamalib_test/test.txt")
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
