import torch 
import json 
import re
from collections import Counter
import yaml

def create_question_vocabulary(data):
    words_list = []
    for keys, values in data.items():
        question = values['question']
        question = re.sub(r'[^\w\s]', '', question.lower())
        words = question.split()
        words_list.extend(words)

    return set(words_list)


if __name__ == '__main__':

    dataset_train_file = '/home/mihnea/data/VQA_DL/miniCOCO/miniCOCOtrain.json'
      
    with open(dataset_train_file) as f:
        data = json.load(f)

    words = create_question_vocabulary(data)

    word_map = {word : idx + 1 for idx, word in enumerate(words)}
    word_map['<unk>'] = len(words) + 1
    word_map['<start>'] = len(words) + 1
    word_map['<end>'] = len(words) + 1
    word_map['<pad>'] = 0

    with open('/home/mihnea/vqa/configs/word_vocabulary.yaml', 'w') as outfile:
        yaml.dump(word_map, outfile)




    

    
