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
from criterion import L1
from torch.utils.tensorboard import SummaryWriter


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
                                  batch_size=16, 
                                  num_workers=4, 
                                  drop_last=True)
    
    validation_dataloader = DataLoader(validation_dataset, 
                                  batch_size=2, 
                                  num_workers=4, 
                                  drop_last=True)

    model = VQA(input_size_text_rnn=cfg.MODEL.TEXT_RNN.INPUT_SIZE, 
                hidden_size_text_rnn=cfg.MODEL.TEXT_RNN.HIDDEN_EMBEDDING_SIZE, 
                vocabulary_size=cfg.DATASET.WORD_VOCABULARY_SIZE, 
                no_answers=train_dataset.len_answers, 
                device = device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    criterion = torch.nn.L1Loss(reduction='sum')

    model.train()

    for epoch in range(cfg.TRAIN.EPOCHS):

        writer.add_scalar("Epoch", epoch)
        for idx_batch, batch in enumerate(train_dataloader):
            img_embedding, text_embeddings, labels, image_name, question_text = batch
            text_embeddings = text_embeddings.to(device)
            
            optimizer.zero_grad()
            logits = model(text_embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Train Loss", loss.item())



            
    


if __name__ == '__main__':
    main()



