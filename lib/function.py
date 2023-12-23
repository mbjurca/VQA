import sys

sys.path.append('../lib/')
sys.path.append('../utils/')

import torch 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import accuracy
from utils.Evaluator import Evaluator
from configs import update_configs, get_configs

writer = SummaryWriter('../runs_0001')

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

            optimizer.step()

            acc = accuracy(logits, labels)
            train_acc.append(acc)
            train_loss.append(loss.item())

            predictions = torch.argmax(logits, dim=1).cpu()
            # save predictions so that we can compute metrics later
            for prediction, question_id in zip(predictions, question_ids):
                result_list.append({
                    "answer": prediction.item(),
                    "question_id": question_id
                })

        #todo: add this to config
        annFile = '../data/miniCOCO_train_annotations.json'
        quesFile = '../data/miniCOCO_train_questions.json'
        evaluator = Evaluator(annFile, quesFile, config.DATASET.IDS_TO_LABELS, result_list)
        evaluator.print_accuracies()

        writer.add_scalar("Epoch", epoch, epoch)
        writer.add_scalar("Train Loss", torch.tensor(train_loss).mean(), epoch)
        writer.add_scalar("Train Accuracy", torch.tensor(train_acc).mean(), epoch)
        print(f'Train Accuracy : {torch.tensor(train_acc).mean()}')


def eval(validation_dataloader, config, model, criterion, device):

    for epoch in range(config.TRAIN.EPOCHS):

        # TODO FIX evaluation 

        model.eval()
        for idx_batch, val_batch in enumerate(validation_dataloader):

            img_embedding, text_embeddings, labels, image_name, question_text = val_batch
            text_embeddings = text_embeddings.to(device)
            
            logits = model(text_embeddings, img_embedding)
            loss = criterion(logits, labels)

            acc = accuracy(logits, labels)

            writer.add_scalar("Epoch", epoch, epoch * len(validation_dataloader) + idx_batch)
            writer.add_scalar("Val Loss", loss.item(), epoch * len(validation_dataloader) + idx_batch)
            writer.add_scalar("Val Accuracy", acc, epoch * len(validation_dataloader) + idx_batch)

