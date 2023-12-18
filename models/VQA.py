import torch
from torch import nn
from text import Text
from image import Image
from LanguageModel import LanguageModel
import torch.nn.functional as F

class VQA(nn.Module):

    def __init__(self, 
                input_size_text_rnn,
                hidden_size_text_rnn,
                no_in_features_vit, 
                no_out_features_vit, 
                no_patches_vit, 
                no_transformer_blocks_vit,
                no_transformer_heads_vit,
                dropout_vit,
                no_features_slm,
                sequence_length_slm,
                no_transformer_blocks_slm,
                no_transformer_heads_slm,
                dropout_slm,
                vocabulary_size,
                no_answers,
                device, 
                config):
        super(VQA, self).__init__()
        self.input_size_text_rnn = input_size_text_rnn
        self.hidden_size_text_rnn = hidden_size_text_rnn
        self.no_answers = no_answers
        self.device = device
        
        self.text_embedding = nn.Embedding(vocabulary_size, input_size_text_rnn)
        self.text_rnn = Text(input_size=input_size_text_rnn, 
                            hidden_size=hidden_size_text_rnn, 
                            device=device,
                            model_type = config.MODEL.TEXT.TYPE)
        
        self.vit = Image(no_in_features=no_in_features_vit,
                         no_out_features=no_out_features_vit,
                         no_patches=no_patches_vit,
                         no_transformer_blocks=no_transformer_blocks_vit,
                         no_heads=no_transformer_heads_vit,
                         dropout=dropout_vit,
                         model_type = config.MODEL.IMAGE.TYPE)

        self.slm = LanguageModel(no_features=no_features_slm,
                       sequence_length=sequence_length_slm,
                       no_transformer_blocks=no_transformer_blocks_slm,
                       no_heads=no_transformer_heads_slm,dropout=dropout_slm,
                       model_type = config.MODEL.LANGUAGE_MODEL.TYPE)

        self.MLP = nn.Sequential(nn.Linear(hidden_size_text_rnn+no_out_features_vit, 1000), 
                                 nn.Tanh(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1000, 1000), 
                                 nn.Tanh(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1000, no_answers)
                                 )

    def forward(self, x_text, x_image):
        
        x_text = self.text_embedding(x_text)
        x_text = self.text_rnn(x_text) # batch_size, no_answers
        x_text = self.slm(x_text)

        x_image = self.vit(x_image)
        x = torch.cat((x_text, x_image), dim=1)
        x = self.MLP(x)

        return x

        
        




