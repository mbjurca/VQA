import sys

sys.path.append('../configs/')
sys.path.append('../datasets/')

import torch 
from torch.utils.data import DataLoader
from configs import update_configs, get_configs
from coco_vqa_dataset import VQA_dataset


DATASET_CFG_FILE = "../configs/dataset.yaml"
MODEL_CFG_FILE = "../configs/model.yaml"
TRAIN_CFG_FILE = "../configs/train.yaml"


def main():
    
    # create the config
    cfg = get_configs()
    update_configs(cfg, MODEL_CFG_FILE, DATASET_CFG_FILE, TRAIN_CFG_FILE)

    train_dataset = VQA_dataset(cfg.DATASET.TRAIN_FILE, cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER)
    validation_dataset = VQA_dataset(cfg.DATASET.VAL_FILE, cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER)


    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=16, 
                                  num_workers=4)
    
    validation_dataloader = DataLoader(validation_dataset, 
                                  batch_size=2, 
                                  num_workers=4)
    
    
    for j, batch in enumerate(train_dataloader):
        print(batch)
            
    


if __name__ == '__main__':
    main()



