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

    train_dataset = VQA_dataset(dataset_file=cfg.DATASET.TRAIN_FILE,
                                labels_file=cfg.DATASET.LABELS,
                                vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER)
    
    validation_dataset = VQA_dataset(dataset_file=cfg.DATASET.VAL_FILE,
                                    labels_file=cfg.DATASET.LABELS,
                                    vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                    image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER)


    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=16, 
                                  num_workers=4)
    
    validation_dataloader = DataLoader(validation_dataset, 
                                  batch_size=2, 
                                  num_workers=4)
    
    
    for j, batch in enumerate(train_dataloader):
        img_embedding, text_embeddings, label, image_name, question_text = batch
        print(img_embedding.shape, text_embeddings.shape, label.shape)

            
    


if __name__ == '__main__':
    main()



