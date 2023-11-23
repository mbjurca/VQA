import torch 
import json 
import re
from collections import Counter
import yaml


def extract_labels(data):

    answers = []
    for _, values in data.items():
        answers.extend(values['answers'])

    answers = set(answers)
    labels = {answer : index for index, answer in enumerate(answers)}

    return labels


if __name__ == '__main__':

    dataset_train_file = '/home/mihnea/data/VQA_DL/miniCOCO/miniCOCOtrain.json'
      
    with open(dataset_train_file) as f:
        data = json.load(f)

    labels = extract_labels(data)

    with open('/home/mihnea/vqa/configs/labels.yaml', 'w') as outfile:
        yaml.dump(labels, outfile)