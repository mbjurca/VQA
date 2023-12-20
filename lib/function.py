import sys

sys.path.append('../lib/')
sys.path.append('../utils/')

import torch 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from metrics import accuracy

writer = SummaryWriter()

def train(train_dataloader, config, model, optimizer, criterion, scheduler, device):

    for epoch in range(config.TRAIN.EPOCHS):

        model.train()
        train_acc = []
        train_loss = []
        for idx_batch, train_batch in enumerate(tqdm(train_dataloader)):
            img_embedding, text_embeddings, labels, image_name, question_text = train_batch
            # print(question_text)
            text_embeddings = text_embeddings.to(device)
            img_embedding = img_embedding.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(text_embeddings, img_embedding)
            # print(logits.shape, labels.shape)
            loss = criterion(logits, labels.float())
            loss.backward()

            clipping_value = .5 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            # scheduler.step()

            acc = accuracy(logits, labels.float())
            train_acc.append(acc)
            train_loss.append(loss.item())

            # Find indices and values where logits > 0.1
            # indices_greater_than_0_1 = (logits > 0.1).nonzero()
            
            # if indices_greater_than_0_1.numel() > 0:
            #     print("Indices and values where logits > 0.1:")
            #     for idx in indices_greater_than_0_1:
            #         row, col = idx[0], idx[1]
            #         value = logits[row, col]
            #         print(f"Index ({row}, {col}): Value {value}")


        writer.add_scalar("Epoch", epoch, epoch)
        writer.add_scalar("Train Loss", torch.tensor(train_loss).mean(), epoch)
        writer.add_scalar("Train Accuracy", torch.tensor(train_acc).mean(), epoch)


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

