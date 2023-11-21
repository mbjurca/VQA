import torch 
from configs import update_configs, get_configs
from torch.utils import data
import json

class VQA_dataset(data.Dataset):

    def __init__(self,
                 dataset_file, 
                 image_file):
        
        # Create vocabulary dictionaries  
        questions_word_list = []
        answers_list = []
        with open(dataset_file) as f:
            data = json.load(f)
    
    def __len__(self):
        return 0
    
    def create_question_vocabulary(self, data_dict):
        pass



