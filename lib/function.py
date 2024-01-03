import sys

sys.path.append('../lib/')
sys.path.append('../utils/')

import torch 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from metrics import accuracy
import torch.nn as nn
import matplotlib.pyplot as plt

writer = SummaryWriter()

def train(train_dataloader, config, model, optimizer, criterion, scheduler, device):

    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        train_acc = []
        train_loss = []

        for idx_batch, train_batch in enumerate(tqdm(train_dataloader)):
            img_embedding, text_embedding, labels, _, _ = train_batch
            text_embedding = text_embedding.to(device)
            img_embedding = img_embedding.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(text_embedding, img_embedding)

            loss = criterion(logits, labels)
            loss.backward()
            train_loss.append(loss.item())

            #clipping_value = .5 # arbitrary value of your choosing
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            # scheduler.step()

            acc = accuracy(logits, labels)
            train_acc.append(acc)

            # Find indices and values where logits > 0.1
            # indices_greater_than_0_1 = (logits > 0.1).nonzero()
            
            # if indices_greater_than_0_1.numel() > 0:
            #     print("Indices and values where logits > 0.1:")
            #     for idx in indices_greater_than_0_1:
            #         row, col = idx[0], idx[1]
            #         value = logits[row, col]
            #         print(f"Index ({row}, {col}): Value {value}")

        print('Epoch {}/{}, Iter {}/{}, Train Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, config.TRAIN.EPOCHS, idx_batch, 
                                                                                       len(train_dataloader),
                                                                                       torch.tensor(train_loss).mean(), 
                                                                                       torch.tensor(train_acc).mean() * 100.))

        writer.add_scalar("Epoch", epoch, epoch)
        writer.add_scalar("Train Loss", torch.tensor(train_loss).mean(), epoch)
        writer.add_scalar("Train Accuracy", torch.tensor(train_acc).mean(), epoch)



def val(validation_dataloader, config, model, criterion, device):

    for epoch in range(config.TRAIN.EPOCHS):

        model.eval()
        eval_acc = []
        eval_loss = []

        for _, val_batch in enumerate(tqdm(validation_dataloader)):

            img_embedding, text_embedding, labels, _, _ = val_batch
            text_embedding = text_embedding.to(device)
            img_embedding = img_embedding.to(device)
            labels = labels.to(device)

            logits = model(text_embedding, img_embedding)
            loss = criterion(logits, labels)
            eval_loss.append(loss.item())

            acc = accuracy(logits, labels)
            eval_acc.append(acc)
            
        writer.add_scalar("Epoch", epoch, epoch * len(validation_dataloader))
        writer.add_scalar("Val Loss", torch.tensor(eval_loss).mean(), epoch * len(validation_dataloader))
        writer.add_scalar("Val Accuracy", torch.tensor(eval_acc).mean(), epoch * len(validation_dataloader))

        print('Epoch {}/{}, Val Loss: {:.3f}, Val Accuracy: {:.3f}'.format(epoch, config.TRAIN.EPOCHS,
                                                                           torch.tensor(eval_loss).mean(), 
                                                                           torch.tensor(eval_acc).mean() * 100.))

