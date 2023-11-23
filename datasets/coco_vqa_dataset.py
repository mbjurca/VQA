import torch 
from configs import update_configs, get_configs
from torch.utils import data
import json
import re
import numpy as np
import os
from collections import Counter
import yaml

class VQA_dataset(data.Dataset):

    def __init__(self,
                 dataset_file, 
                 labels_file,
                 vocabulary_file,
                 image_embedding_folder, 
                 max_len_text_embedding=15):
        
        self.image_embedding_folder = image_embedding_folder
        self.max_len_text_embedding = max_len_text_embedding
        
        with open(dataset_file) as f:
            self.data = json.load(f)

        self.labels_dict = self.load_labels(labels_file)
        self.len_answers = len(self.labels_dict.keys())

        self.word_dict = self.load_vocabulary(vocabulary_file)
        self.len_word_dict = len(self.word_dict.keys())


    def generate_sample(self, index):

        question_id = list(self.data)[index]

        question_text = self.data[question_id]['question']
        image_name = self.data[question_id]['image_name']
        answers = self.data[question_id]['answers']
        image_id = self.data[question_id]['image_id']

        # get image embedding
        img_embedding = self.get_image_embedding(image_id)
        text_embeddings = torch.tensor(self.get_text_embeddings(question_text))
        label = self.compute_label(answers)

        return img_embedding, text_embeddings, label, image_name, question_text
    
    def compute_label(self, answers):

        label = np.zeros(self.len_answers)
        set_answers = set(answers)

        for answer in set_answers:
            count = answers.count(answer)
            print(self.labels_dict[answer], answer, count, self.len_answers)
            if count >= 4:
                label[self.labels_dict[answer]] = 1.
            elif count == 3:
                label[self.labels_dict[answer]] = 0.9
            elif count == 2:
                label[self.labels_dict[answer]] = 0.6
            elif count == 1:
                label[self.labels_dict[answer]] = 0.3

        return label

    def get_image_embedding(self, image_id):

        features = np.load(os.path.join(self.image_embedding_folder, f'{image_id}.npz'))['feat']

        return features
    
    def get_text_embeddings(self, question_text):

        question = re.sub(r'[^\w\s]', '', question_text.lower())
        words = question.split()
        words_token = [self.word_dict.get(word, self.word_dict['<unk>']) for word in words]

        embedding = [self.word_dict['<start>']] + words_token[:self.max_len_text_embedding-2] + \
                    [self.word_dict['<pad>']] * (self.max_len_text_embedding - len(words_token) - 2) + \
                    [self.word_dict['<end>']]

        return embedding

    def load_labels(self, labels_file):

        with open(labels_file) as f:
            labels = yaml.safe_load(f)

        return labels
    
    def load_vocabulary(self, vocabulary_file):

        with open(vocabulary_file) as f:
            vocabulary = yaml.safe_load(f)

        return vocabulary

    
    def __len__(self):
        return len(list(self.data))

    def __getitem__(self, index):

        img_embedding, text_embeddings, label, image_name, question_text = self.generate_sample(index)

        return  img_embedding, text_embeddings, label, image_name, question_text


