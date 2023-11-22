import sys

sys.path.append('../configs/')
sys.path.append('../datasets/')

from configs import update_configs, get_configs
from coco_vqa_dataset import VQA_dataset

print(sys.path)