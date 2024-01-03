import sys

sys.path.append('../lib/')
sys.path.append('../utils/')

import torch 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from metrics import accuracy
from utils.Evaluator import Evaluator
from configs import update_configs, get_configs
import torch.nn as nn
import matplotlib.pyplot as plt

writer = SummaryWriter('../runs_complex_0001')

def train(train_dataloader, config, model, optimizer, criterion, device):

    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        train_acc = []
        train_loss = []
        result_list = []

        for idx_batch, train_batch in enumerate(tqdm(train_dataloader)):
            img_embedding, text_embeddings, labels, image_name, question_text, question_ids = train_batch
            text_embeddings = text_embeddings.to(device)
            img_embedding = img_embedding.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(text_embeddings, img_embedding)
            loss = criterion(logits, labels)
            loss.backward()
            train_loss.append(loss.item())

            #clipping_value = .5 # arbitrary value of your choosing
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            # scheduler.step()

            acc = accuracy(logits, labels)
            train_acc.append(acc)

            predictions = torch.argmax(logits, dim=1).cpu()
            # save predictions so that we can compute metrics later
            for prediction, question_id in zip(predictions, question_ids):
                result_list.append({
                    "answer": prediction.item(),
                    "question_id": question_id
                })

        print('Epoch {}/{}, Iter {}/{}, Train Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, config.TRAIN.EPOCHS, idx_batch,
                                                                                       len(train_dataloader),
                                                                                       torch.tensor(train_loss).mean(),
                                                       torch.tensor(train_acc).mean() * 100.))

        evaluator = Evaluator(config.TRAIN.ANNOTATIONS_FILE, config.TRAIN.QUESTIONS_FILE, config.DATASET.IDS_TO_LABELS, result_list)
        acc = evaluator.get_overall_accuracy()
        writer.add_scalar("Epoch", epoch, epoch)
        writer.add_scalar("Train Loss", torch.tensor(train_loss).mean(), epoch)
        writer.add_scalar("Train old Accuracy", torch.tensor(train_acc).mean(), epoch)
        writer.add_scalar("Train toolkit Accuracy", acc, epoch)
        print(f'Train old Accuracy : {torch.tensor(train_acc).mean()}')
        print(f'Train toolkit Accuracy : {acc}')



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

