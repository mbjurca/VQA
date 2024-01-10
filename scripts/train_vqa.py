import sys

sys.path.append('../configs/')
sys.path.append('../datasets/')
sys.path.append('../models/')
sys.path.append('../lib/')

import torch 
from torch.utils.data import DataLoader, SequentialSampler
from configs import update_configs, get_configs
from coco_vqa_dataset import VQA_dataset
from VQA import VQA
from function import train
from torch.optim.lr_scheduler import StepLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import WarmupThenScheduler

DATASET_CFG_FILE = "../configs/dataset.yaml"
MODEL_CFG_FILE = "../configs/model.yaml"
TRAIN_CFG_FILE = "../configs/train.yaml"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment')
    args = parser.parse_args()

    summary_writer_path = f'../experiments/{args.exp_name}'

    # set device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the config
    cfg = get_configs()
    update_configs(cfg, MODEL_CFG_FILE, DATASET_CFG_FILE, TRAIN_CFG_FILE)

    train_dataset = VQA_dataset(dataset_file=cfg.DATASET.TRAIN_FILE,
                                labels_to_ids_file=cfg.DATASET.TRAIN_LABELS_TO_IDS,
                                ids_to_labels_file=cfg.DATASET.TRAIN_IDS_TO_LABELS,
                                vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER,
                                token_type = cfg.MODEL.TEXT.TOKEN_TYPE)

    train_dataloader = DataLoader(train_dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
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
                no_features_slm=cfg.MODEL.LANGUAGE_MODEL.NO_FEATURES,
                sequence_length_slm=cfg.MODEL.LANGUAGE_MODEL.SEQUENCE_LENGTH,
                no_transformer_blocks_slm=cfg.MODEL.LANGUAGE_MODEL.NO_BLOCKS,
                no_transformer_heads_slm=cfg.MODEL.LANGUAGE_MODEL.NO_HEADS,
                dropout_slm=cfg.MODEL.LANGUAGE_MODEL.DROPOUT,
                vocabulary_size=cfg.DATASET.WORD_VOCABULARY_SIZE, 
                no_answers=train_dataset.len_answers, 
                device = device,
                config = cfg).to(device)
    
    match cfg.TRAIN.OPTIMIZER:
        case 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=(cfg.TRAIN.ADAM.BETA_1, cfg.TRAIN.ADAM.BETA_2))
    
    match cfg.TRAIN.CRITERION:
        case 'bce':
            criterion = torch.nn.BCELoss(reduction='sum')

    match cfg.TRAIN.LR_SCHEDULER:
        case 'step':     
            step_sheduler = StepLR(optimizer, step_size=cfg.TRAIN.STEP_LR.STEP_SIZE, gamma=cfg.TRAIN.STEP_LR.GAMMA)
            scheduler_lr = WarmupThenScheduler(optimizer=optimizer,
                                            warmup_epochs=cfg.TRAIN.WARMUP,
                                            after_warmup_scheduler=step_sheduler,
                                            final_lr=cfg.TRAIN.LR)
            
    grad_norm = cfg.TRAIN.GRAD_CLIP_NORM if cfg.TRAIN.GRAD_CLIP_NORM != None else None

    validation_dataset = VQA_dataset(dataset_file=cfg.DATASET.VAL_FILE,
                                     labels_to_ids_file=cfg.DATASET.TRAIN_LABELS_TO_IDS,
                                     ids_to_labels_file=cfg.DATASET.TRAIN_IDS_TO_LABELS,
                                     vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                     image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER,
                                     token_type=cfg.MODEL.TEXT.TOKEN_TYPE)

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=cfg.VAL.BATCH_SIZE,
                                       num_workers=4,
                                       shuffle=True)

    writer = SummaryWriter(summary_writer_path)

    train(train_dataloader = train_dataloader,
          validation_dataloader = validation_dataloader,
          config = cfg, 
          model = model,
          optimizer = optimizer,
          grad_norm = grad_norm,
          criterion = criterion,
          scheduler_lr = scheduler_lr,
          device = device, 
          writer=writer, 
          )


if __name__ == '__main__':
    main()

