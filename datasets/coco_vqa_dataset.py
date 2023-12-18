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
                 max_len_text_embedding=15,
                 token_type = "START_END"):
        
        self.image_embedding_folder = image_embedding_folder
        self.max_len_text_embedding = max_len_text_embedding
        
        with open(dataset_file) as f:
            self.data = json.load(f)

        if not(os.path.isfile(labels_file)):
            self.generate_labels(labels_file)
            
        self.labels_dict = self.load_labels(labels_file)
        self.len_answers = len(self.labels_dict.keys())

        if not(os.path.isfile(vocabulary_file)):
            self.generate_vocabulary(vocabulary_file)

        self.word_dict = self.load_vocabulary(vocabulary_file)
        self.len_word_dict = len(self.word_dict.keys())
        self.token_type = token_type


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
            count = int(answers.count(answer))
            if count >= 4:
                label[self.labels_dict.get(answer, 0)] = 1.
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

        if self.token_type == "START_END":
            embedding = [self.word_dict['<start>']] + words_token[:self.max_len_text_embedding-2] + \
                        [self.word_dict['<pad>']] * (self.max_len_text_embedding - len(words_token) - 2) + \
                        [self.word_dict['<end>']]
            
        elif self.token_type == "PAD_ONLY":
            embedding = words_token[:self.max_len_text_embedding] + [self.word_dict['<pad>']] * (self.max_len_text_embedding - len(words_token))

        return embedding

    def extract_labels(self):

        answers = []
        for _, values in self.data.items():
            answers.extend(values['answers'])

        answers = set(answers)
        labels = {answer : index for index, answer in enumerate(answers)}

        return labels
    
    def generate_labels(self, labels_file):

        labels = self.extract_labels()

        with open(labels_file, 'w') as outfile:
            yaml.dump(labels, outfile)

    def load_labels(self, labels_file):

        with open(labels_file) as f:
            labels = yaml.safe_load(f)

        return labels
    
    def create_vocabulary(self):

        words_list = []
        for _, values in self.data.items():
            question = values['question']
            question = re.sub(r'[^\w\s]', '', question.lower())
            words = question.split()
            words_list.extend(words)

        return set(words_list)
    
    def generate_vocabulary(self, vocabulary_file):

        words = self.create_vocabulary()
          
        word_map = {word : idx + 1 for idx, word in enumerate(words)}
        word_map['<unk>'] = len(words) + 1
        word_map['<start>'] = len(words) + 1
        word_map['<end>'] = len(words) + 1
        word_map['<pad>'] = 0

        with open(vocabulary_file, 'w') as outfile:
            yaml.dump(word_map, outfile)

    def load_vocabulary(self, vocabulary_file):

        with open(vocabulary_file) as f:
            vocabulary = yaml.safe_load(f)

        return vocabulary
    
    def __len__(self):
        return 23045

    def __getitem__(self, index):

        img_embedding, text_embeddings, label, image_name, question_text = self.generate_sample(index)

        return  img_embedding, text_embeddings, label, image_name, question_text

