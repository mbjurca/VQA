import torch
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
                 labels_to_ids_file,
                 ids_to_labels_file,
                 vocabulary_file,
                 image_embedding_folder,
                 max_len_text_embedding=15,
                 token_type = "START_END"):
        
        self.image_embedding_folder = image_embedding_folder
        self.max_len_text_embedding = max_len_text_embedding
        
        # load dataset
        with open(dataset_file) as f:
            self.data = json.load(f)

        # D: ar trebui sa le generam mereu pentru ca difera pe train/validation sau sa lasam un comment in README ca nu trebuie
        # sa existe inainte de train/val
        # generate labels for answers
        if not(os.path.isfile(labels_to_ids_file)) or not(os.path.isfile(ids_to_labels_file)):
            self.generate_labels(labels_to_ids_file, ids_to_labels_file)

        self.labels_dict = self.load_labels(labels_to_ids_file)
        self.len_answers = len(self.labels_dict.keys())

        # D: ar trebui sa il generam mereu pentru ca difera pe train/validation sau sa lasam un comment in README ca nu trebuie
        # sa existe inainte de train/val
        # generate word vocabulary
        if not(os.path.isfile(vocabulary_file)):
            self.generate_vocabulary(vocabulary_file)

        self.word_dict = self.load_vocabulary(vocabulary_file)
        self.len_word_dict = len(self.word_dict.keys())
        self.token_type = token_type

        #remove entries that have labels unknown to the trained model
        self.clean_dataset()


    def generate_sample(self, index):

        # each sample is defined by an unique question id
        question_id = list(self.data)[index]

        question_text = self.data[question_id]['question']
        image_name = self.data[question_id]['image_name']
        answers = self.data[question_id]['answers']
        image_id = self.data[question_id]['image_id']

        # get image embedding
        img_embedding = self.get_image_embedding(image_id)
        # get text embedding
        text_embeddings = torch.tensor(self.get_text_embeddings(question_text))
        # compute labels for answers
        label = self.compute_label(answers)

        # if np.sum(label) == 0:

        return img_embedding, text_embeddings, label, image_name, question_text, question_id
    
    def compute_label(self, answers):

        # compute labels based on the number of occurrences for each answer
        label = np.zeros(self.len_answers)
        set_answers = set(answers)
        for answer in set_answers:
            count = int(answers.count(answer))
            if self.labels_dict.get(answer, None) != None:
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

        # load pre-computed features for image
        features = np.load(os.path.join(self.image_embedding_folder, f'{image_id}.npz'))['feat']

        return features
    
    def get_text_embeddings(self, question_text):

        # split the question into lower-case words and map it using the word vocabulary
        question = re.sub(r'[^\w\s]', '', question_text.lower())
        words = question.split()
        words_token = [self.word_dict.get(word, self.word_dict['<unk>']) for word in words]

        # a question is tokenized with start-end tokens and padding
        if self.token_type == "START_END":
            embedding = [self.word_dict['<start>']] + words_token[:self.max_len_text_embedding-2] + \
                        [self.word_dict['<pad>']] * (self.max_len_text_embedding - len(words_token) - 2) + \
                        [self.word_dict['<end>']]
        # a question is only padded
        elif self.token_type == "PAD_ONLY":
            embedding = words_token[:self.max_len_text_embedding] + [self.word_dict['<pad>']] * (self.max_len_text_embedding - len(words_token))

        return embedding

    def extract_labels(self):

        answers = []
        for _, values in self.data.items():
            answers.extend(values['answers'])

        answers = set(answers)
        labels_to_ids = {answer: index for index, answer in enumerate(answers)}
        ids_to_labels = {index: answer for index, answer in enumerate(answers)}

        return labels_to_ids, ids_to_labels
    
    def generate_labels(self, labels_to_ids_file, ids_to_labels_file):

        labels_to_ids, ids_to_labels = self.extract_labels()

        with open(labels_to_ids_file, 'w') as outfile:
            yaml.dump(labels_to_ids, outfile)
        with open(ids_to_labels_file, 'w') as outfile:
            yaml.dump(ids_to_labels, outfile)

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
        # define specific tokens for unknown words, start and end of a question and padding token
        word_map['<unk>'] = len(words) + 1
        word_map['<start>'] = len(words) + 2
        word_map['<end>'] = len(words) + 3
        word_map['<pad>'] = 0

        with open(vocabulary_file, 'w') as outfile:
            yaml.dump(word_map, outfile)

    def load_vocabulary(self, vocabulary_file):

        with open(vocabulary_file) as f:
            vocabulary = yaml.safe_load(f)

        return vocabulary

    def get_top_k_answers(self, k):
        answers_list = []
        for keys, values in self.data.items():
            answers = values['answers']
            answers_list.extend(answers)

        answers_freq = Counter(answers_list)
        most_common_answers = dict(answers_freq.most_common(k))

        return most_common_answers

    def get_question_ids(self):
        return list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_embedding, text_embeddings, label, image_name, question_text, question_id = self.generate_sample(index)

        return  img_embedding, text_embeddings, label, image_name, question_text, question_id


    """
    remove samples that have only answers that DO NOT appear in the train labels set
    """
    def clean_dataset(self):
        filtered_dataset = {}
        allowed_answers = list(self.labels_dict.keys())
        for key, entry in self.data.items():
            # Check if all answers are in the allowed list
            if any(answer in allowed_answers for answer in entry['answers']):
                filtered_dataset[key] = entry
        self.data = filtered_dataset

        print(len(self.data))
