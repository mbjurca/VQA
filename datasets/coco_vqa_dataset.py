import torch 
from configs import update_configs, get_configs
from torch.utils import data
import json
import re
import numpy as np
import os
from collections import Counter

class VQA_dataset(data.Dataset):

    def __init__(self,
                 dataset_file, 
                 image_embedding_folder, 
                 min_word_frequancy):
        
        self.image_embedding_folder = image_embedding_folder
        
        with open(dataset_file) as f:
            self.data = json.load(f)

        self.labels_dict = self.labels2answers(self.data)
        self.len_answers = len(self.labels_dict.keys())

        self.word_dict = self.create_question_vocabulary(self.data)
        self.len_word_dict = len(self.word_dict.keys())


    def generate_sample(self, index):

        question_id = list(self.data)[index]

        question_text = self.data[question_id]['question']
        image_name = self.data[question_id]['image_name']
        answers = self.data[question_id]['answers']
        image_id = self.data[question_id]['image_id']

        # get image embedding
        img_embedding = self.get_image_embedding(image_id)
        text_embeddings = self.get_text_embeddings(question_text)
        label = self.compute_label(answers)

        print(img_embedding.shape, text_embeddings.shape, label.shape)

        return img_embedding, label
    
    def compute_label(self, answers):

        label = np.zeros((self.len_answers))
        set_answers = set(answers)

        for answer in set_answers:
            count = answers.count(answer)
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
        words_idx = [self.word_dict[word] for word in words]

        embeddings = np.zeros((len(words), self.len_word_dict))
        embeddings[:, words_idx] = 1

        return embeddings

    def labels2answers(self, data):

        answers = []
        for _, values in data.items():
            answers.extend(values['answers'])

        answers = set(answers)
        labels = {answer : index for index, answer in enumerate(answers)}

        return labels
    
    def create_question_vocabulary(self, data):
        words_list = []
        for keys, values in data.items():
            question = values['question']
            question = re.sub(r'[^\w\s]', '', question.lower())
            words = question.split()
            words_list.extend(words)

        words = set(words_list)
        labels = {word : index for index, word in enumerate(words)}

        return labels
    
    def __len__(self):
        return len(list(self.data))

    def __getitem__(self, index):

        img_embedding, label = self.generate_sample(index)

        return  img_embedding, label


