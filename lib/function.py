import sys

sys.path.append('../utils/')

import torch
from tqdm import tqdm
from metrics import accuracy
from Evaluator import Evaluator
from configs import update_configs, get_configs
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

def train(train_dataloader, validation_dataloader, config, model, optimizer, grad_norm, criterion, scheduler_lr, device, writer):
    top_accuracies = []  # List to store the top accuracies
    top_models = []  # List to store the state dicts of the top models
    best_acc, best_loss = 0, float('inf')
    no_improve_counter = 0
    n_epochs_stop = config.TRAIN.EARLY_STOPPING_EPOCHS_NUM
    performance_improvement_eps = config.TRAIN.EARLY_STOPPING_PERFORMANCE_EPS

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
            loss = criterion(logits, labels.float())

            loss.backward()
            train_loss.append(loss.item())
            
            if grad_norm != None:
                clip_grad_norm_(model.parameters(), grad_norm)  # You can adjust the value (3.0) as needed

            optimizer.step()

            acc = accuracy(logits, labels)
            train_acc.append(acc)

            predictions = torch.argmax(logits, dim=1).cpu()
            # save predictions so that we can compute metrics later
            for prediction, question_id in zip(predictions, question_ids):
                result_list.append({
                    "answer": prediction.item(),
                    "question_id": question_id
                })
        
        scheduler_lr.step()

        print('Epoch {}/{}, Iter {}/{}, Train Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, config.TRAIN.EPOCHS, idx_batch,
                                                                                           len(train_dataloader),
                                                                                           torch.tensor(train_loss).mean(),
                                                                                           torch.tensor(train_acc).mean() * 100.))

        evaluator = Evaluator(config.TRAIN.ANNOTATIONS_FILE, config.TRAIN.QUESTIONS_FILE, config.DATASET.TRAIN_IDS_TO_LABELS, result_list)
        acc = evaluator.get_overall_accuracy()
        writer.add_scalar("Epoch", epoch, epoch)
        writer.add_scalar("Train Loss", torch.tensor(train_loss).mean(), epoch)
        writer.add_scalar("Train old Accuracy", torch.tensor(train_acc).mean(), epoch)
        writer.add_scalar("Train toolkit Accuracy", acc, epoch)
        print(f'Train old Accuracy : {torch.tensor(train_acc).mean()}')
        print(f'Train toolkit Accuracy : {acc}')
        val_acc = val(validation_dataloader=validation_dataloader,
                      config = config,
                      model=model,
                      criterion=criterion,
                      device=device,
                      writer=writer,
                      epoch=epoch)

        if val_acc > best_acc + performance_improvement_eps:
            best_acc = max(val_acc, best_acc)
            no_improve_counter = 0
            # Save model if it's one of the top 3
            if len(top_accuracies) < 3:
                top_accuracies.append(val_acc)
                top_models.append(model.state_dict())
            else:
                # Find the position to insert the new accuracy
                insert_pos = next((i for i, acc in enumerate(top_accuracies) if val_acc > acc), len(top_accuracies))

                # Insert and maintain size
                top_accuracies.insert(insert_pos, val_acc)
                top_models.insert(insert_pos, model.state_dict())

                if len(top_accuracies) > 3:
                    # Keep only top 3
                    top_accuracies = top_accuracies[:3]
                    top_models = top_models[:3]
        else:
            no_improve_counter += 1

        if no_improve_counter >= n_epochs_stop:
            print(f'Early stopping triggered after epoch {epoch}')
            break


    for i, state_dict in enumerate(top_models):
        torch.save(state_dict, f'../saved_models/top_model_{i + 1}_acc_{top_accuracies[i]}.pth')

def val(validation_dataloader, config, model, criterion, device, writer, epoch=None):

    model.eval()
    eval_acc = []
    eval_loss = []
    result_list = []
    print(f'Running validation\n')

    for _, val_batch in enumerate(tqdm(validation_dataloader)):

        img_embedding, text_embeddings, labels, image_name, question_text, question_ids = val_batch

        text_embedding = text_embeddings.to(device)
        img_embedding = img_embedding.to(device)
        labels = labels.to(device)

        logits = model(text_embedding, img_embedding)
        # max_probabilities = torch.max(torch.nn.functional.softmax(logits), dim=1)
        # print(max_probabilities)
        loss = criterion(logits, labels.float())
        eval_loss.append(loss.item())

        predictions = torch.argmax(logits, dim=1).cpu()
        # save predictions so that we can compute metrics later
        for prediction, question_id in zip(predictions, question_ids):
            result_list.append({
                "answer": prediction.item(),
                "question_id": question_id
            })
        acc = accuracy(logits, labels)
        eval_acc.append(acc)

    evaluator = Evaluator(config.VAL.ANNOTATIONS_FILE, config.VAL.QUESTIONS_FILE, config.DATASET.TRAIN_IDS_TO_LABELS, result_list)
    acc = evaluator.get_overall_accuracy()
    if epoch != None:
        writer.add_scalar("Val Loss", torch.tensor(eval_loss).mean(), epoch)
        writer.add_scalar("Val Accuracy", torch.tensor(eval_acc).mean(), epoch)

        print('Epoch {}/{}, Val Loss: {:.3f}, Val Accuracy: {:.3f}'.format(epoch, config.TRAIN.EPOCHS,
                                                                           torch.tensor(eval_loss).mean(),
                                                                           torch.tensor(eval_acc).mean() * 100.))
        writer.add_scalar("Val toolkit Accuracy", acc, epoch)
    # print(f'Train old Accuracy : {torch.tensor(train_acc).mean()}')
    print(f'Val toolkit Accuracy : {acc}')
    return acc
