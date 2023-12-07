import sys

sys.path.append('../configs/')
sys.path.append('../datasets/')
sys.path.append('../models/')
sys.path.append('../lib/')
sys.path.append('../utils/')

import torch 
from torch.utils.data import DataLoader
from configs import update_configs, get_configs
from coco_vqa_dataset import VQA_dataset
from VQA import VQA
from criterion import L1
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from metrics import accuracy
from ViT import ViT
import torch.nn as nn

DATASET_CFG_FILE = "../configs/dataset.yaml"
MODEL_CFG_FILE = "../configs/model.yaml"
TRAIN_CFG_FILE = "../configs/train.yaml"


def main():
    # set device 

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'

    # create the config
    cfg = get_configs()
    update_configs(cfg, MODEL_CFG_FILE, DATASET_CFG_FILE, TRAIN_CFG_FILE)

    writer = SummaryWriter()

    train_dataset = VQA_dataset(dataset_file=cfg.DATASET.TRAIN_FILE,
                                labels_file=cfg.DATASET.LABELS,
                                vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER)
    
    validation_dataset = VQA_dataset(dataset_file=cfg.DATASET.VAL_FILE,
                                    labels_file=cfg.DATASET.LABELS,
                                    vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                    image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER)


    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=512, 
                                  num_workers=4, 
                                  shuffle=True)
    
    validation_dataloader = DataLoader(validation_dataset, 
                                  batch_size=2, 
                                  num_workers=4, 
                                  shuffle=True)

    model = VQA(input_size_text_rnn=cfg.MODEL.TEXT_RNN.INPUT_SIZE, 
                hidden_size_text_rnn=cfg.MODEL.TEXT_RNN.HIDDEN_EMBEDDING_SIZE, 
                no_in_features_vit=cfg.MODEL.VIT.NO_IN_FEATURES, 
                no_out_features_vit=cfg.MODEL.VIT.NO_OUT_FEATURES, 
                no_patches_vit=cfg.MODEL.VIT.NO_PATCHES, 
                no_transformer_blocks_vit=cfg.MODEL.VIT.NO_BLOCKS,
                no_transformer_heads_vit=cfg.MODEL.VIT.NO_HEADS,
                dropout_vit=cfg.MODEL.VIT.DROPOUT,
                no_features_slm=cfg.MODEL.SLM.NO_FEATURES,
                sequence_length_slm=cfg.MODEL.SLM.SEQUENCE_LENGTH,
                no_transformer_blocks_slm=cfg.MODEL.SLM.NO_BLOCKS,
                no_transformer_heads_slm=cfg.MODEL.SLM.NO_HEADS,
                dropout_slm=cfg.MODEL.SLM.DROPOUT,
                vocabulary_size=cfg.DATASET.WORD_VOCABULARY_SIZE, 
                no_answers=train_dataset.len_answers, 
                device = device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    criterion = torch.nn.L1Loss(reduction='mean')

    for epoch in range(cfg.TRAIN.EPOCHS):

        model.train()
        for idx_batch, train_batch in enumerate(tqdm(train_dataloader)):
            img_embedding, text_embeddings, labels, image_name, question_text = train_batch
            text_embeddings = text_embeddings.to(device)
            
            optimizer.zero_grad()
            logits = model(text_embeddings, img_embedding)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            acc = accuracy(logits, labels)

            writer.add_scalar("Epoch", epoch, epoch * len(train_dataloader) + idx_batch)
            writer.add_scalar("Train Loss", loss.item(), epoch * len(train_dataloader) + idx_batch)
            writer.add_scalar("Train Accuracy", acc, epoch * len(train_dataloader) + idx_batch)

        # model.eval()
        # for idx_batch, val_batch in enumerate(tqdm(validation_dataloader)):

        #     img_embedding, text_embeddings, labels, image_name, question_text = val_batch
        #     text_embeddings = text_embeddings.to(device)
            
        #     logits = model(text_embeddings, img_embedding)
        #     loss = criterion(logits, labels)

        #     acc = accuracy(logits, labels)

        #     writer.add_scalar("Epoch", epoch, epoch * len(validation_dataloader) + idx_batch)
        #     writer.add_scalar("Val Loss", loss.item(), epoch * len(validation_dataloader) + idx_batch)
        #     writer.add_scalar("Val Accuracy", acc, epoch * len(validation_dataloader) + idx_batch)




            
    


if __name__ == '__main__':
    main()



