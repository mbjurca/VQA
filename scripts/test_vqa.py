import sys

sys.path.append('../configs/')
sys.path.append('../datasets/')
sys.path.append('../models/')
sys.path.append('../lib/')

import torch 
from torch.utils.data import DataLoader
from configs import update_configs, get_configs
from coco_vqa_dataset import VQA_dataset
from VQA import VQA
from function import eval


DATASET_CFG_FILE = "../configs/dataset.yaml"
MODEL_CFG_FILE = "../configs/model.yaml"
TRAIN_CFG_FILE = "../configs/train.yaml"


def main():

    # set device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the config
    cfg = get_configs()
    update_configs(cfg, MODEL_CFG_FILE, DATASET_CFG_FILE, TRAIN_CFG_FILE)

    validation_dataset = VQA_dataset(dataset_file=cfg.DATASET.VAL_FILE,
                                    labels_file=cfg.DATASET.LABELS,
                                    vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                    image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER,
                                    token_type = cfg.MODEL.TEXT.TOKEN_TYPE)
    
    validation_dataloader = DataLoader(validation_dataset, 
                                  batch_size=2, 
                                  num_workers=4, 
                                  shuffle=True)

    model = VQA(input_size_text_rnn=cfg.MODEL.TEXT.INPUT_SIZE,
                hidden_size_text_rnn=cfg.MODEL.TEXT.HIDDEN_EMBEDDING_SIZE,
                no_in_features_vit=cfg.MODEL.IMAGE.NO_IN_FEATURES,
                no_out_features_vit=cfg.MODEL.IMAGE.NO_OUT_FEATURES,
                no_patches_vit=cfg.MODEL.IMAGE.NO_PATCHES,
                no_transformer_blocks_vit=cfg.MODEL.IMAGE.NO_BLOCKS,
                no_transformer_heads_vit=cfg.MODEL.IMAGE.NO_HEADS,
                dropout_vit=cfg.MODEL.IMAGE.DROPOUT,
                no_features_slm=cfg.MODEL.MLP.NO_FEATURES,
                sequence_length_slm=cfg.MODEL.MLP.SEQUENCE_LENGTH,
                no_transformer_blocks_slm=cfg.MODEL.MLP.NO_BLOCKS,
                no_transformer_heads_slm=cfg.MODEL.MLP.NO_HEADS,
                dropout_slm=cfg.MODEL.MLP.DROPOUT,
                vocabulary_size=cfg.DATASET.WORD_VOCABULARY_SIZE, 
                no_answers=validation_dataset.len_answers, 
                device = device,
                config = cfg).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    criterion = torch.nn.L1Loss(reduction='mean')

    eval(validation_dataloader = validation_dataloader, 
         config = cfg, 
         model = model, 
         criterion = criterion, 
         device = device)


if __name__ == '__main__':
    main()

