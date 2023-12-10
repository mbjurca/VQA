import sys

sys.path.append('../lib/')
sys.path.append('../utils/')

import torch 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from metrics import accuracy

writer = SummaryWriter()

def train(train_dataloader, config, model, optimizer, criterion, device):

    for epoch in range(config.TRAIN.EPOCHS):

        model.train()
        for idx_batch, train_batch in enumerate(tqdm(train_dataloader)):
            img_embedding, text_embeddings, labels, image_name, question_text = train_batch
            text_embeddings = text_embeddings.to(device)
            img_embedding = img_embedding.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(text_embeddings, img_embedding)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            acc = accuracy(logits, labels)

            writer.add_scalar("Epoch", epoch, epoch * len(train_dataloader) + idx_batch)
            writer.add_scalar("Train Loss", loss.item(), epoch * len(train_dataloader) + idx_batch)
            writer.add_scalar("Train Accuracy", acc, epoch * len(train_dataloader) + idx_batch)


def eval(validation_dataloader, config, model, criterion, device):

    for epoch in range(config.TRAIN.EPOCHS):

        # TODO FIX evaluation 

        model.eval()
        for idx_batch, val_batch in enumerate(tqdm(validation_dataloader)):

            img_embedding, text_embeddings, labels, image_name, question_text = val_batch
            text_embeddings = text_embeddings.to(device)
            
            logits = model(text_embeddings, img_embedding)
            loss = criterion(logits, labels)

            acc = accuracy(logits, labels)

            writer.add_scalar("Epoch", epoch, epoch * len(validation_dataloader) + idx_batch)
            writer.add_scalar("Val Loss", loss.item(), epoch * len(validation_dataloader) + idx_batch)
            writer.add_scalar("Val Accuracy", acc, epoch * len(validation_dataloader) + idx_batch)

