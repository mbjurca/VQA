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

    train_accuracies = []
    train_losses = []

    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        correct = 0
        total = 0

        for idx_batch, train_batch in enumerate(tqdm(train_dataloader)):
            img_embedding, text_embeddings, labels, image_name, question_text = train_batch
            # print(question_text)
            text_embeddings = text_embeddings.to(device)
            img_embedding = img_embedding.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(text_embeddings, img_embedding)
            # print(logits.shape, labels.shape)

            loss = criterion(logits, labels)
            loss.backward()

            #clipping_value = .5 # arbitrary value of your choosing
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            # scheduler.step()

            #epoch_acc = accuracy(logits, labels.float())
            #train_accuracies.append(epoch_acc)

            total += labels.size(0)
            correct += torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1))
            epoch_loss += loss.item()

            # Find indices and values where logits > 0.1
            # indices_greater_than_0_1 = (logits > 0.1).nonzero()
            
            # if indices_greater_than_0_1.numel() > 0:
            #     print("Indices and values where logits > 0.1:")
            #     for idx in indices_greater_than_0_1:
            #         row, col = idx[0], idx[1]
            #         value = logits[row, col]
            #         print(f"Index ({row}, {col}): Value {value}")


        epoch_loss = epoch_loss / len(train_dataloader)
        epoch_acc = 100.* correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print('Epoch {}/{}, Iter {}/{}, Train Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, config.TRAIN.EPOCHS, idx_batch, 
                                                                                       len(train_dataloader),
                                                                                       epoch_loss, 
                                                                                       epoch_acc))

        #writer.add_scalar("Epoch", epoch, epoch)
        #writer.add_scalar("Train Loss", torch.tensor(train_loss).mean(), epoch)
        #writer.add_scalar("Train Accuracy", torch.tensor(train_acc).mean(), epoch)
            
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(torch.tensor(train_losses).cpu().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(torch.tensor(train_accuracies).cpu().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')

    plt.show()


def eval(validation_dataloader, config, model, criterion, device):
    eval_accuracies = []
    eval_losses = []

    for epoch in range(config.TRAIN.EPOCHS):

        epoch_loss = 0
        epoch_acc = 0

        correct = 0
        total = 0

        model.eval()
        for idx_batch, val_batch in enumerate(tqdm(validation_dataloader)):

            img_embedding, text_embeddings, labels, image_name, question_text = val_batch
            text_embeddings = text_embeddings.to(device)
            img_embedding = img_embedding.to(device)
            labels = labels.to(device)

            logits = model(text_embeddings, img_embedding)
            loss = criterion(logits, labels)

            #acc = accuracy(logits, labels)

            total += labels.size(0)
            correct += torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1))
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(validation_dataloader)
        epoch_acc = 100.* correct / total

        writer.add_scalar("Epoch", epoch, epoch * len(validation_dataloader) + idx_batch)
        writer.add_scalar("Val Loss", epoch_loss, epoch * len(validation_dataloader) + idx_batch)
        writer.add_scalar("Val Accuracy", epoch_acc, epoch * len(validation_dataloader) + idx_batch)

        eval_losses.append(epoch_loss)
        eval_accuracies.append(epoch_acc)

        print('Epoch {}/{}, Iter {}/{}, Train Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, config.TRAIN.EPOCHS, idx_batch, 
                                                                                       len(validation_dataloader),
                                                                                       epoch_loss, 
                                                                                       epoch_acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(torch.tensor(eval_losses).cpu().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(torch.tensor(eval_accuracies).cpu().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')

    plt.show()
