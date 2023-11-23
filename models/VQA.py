import torch
from torch import nn
from LSTM import LSTM


class VQA(nn.Module):

    def __init__(self, 
                input_size_text_rnn,
                hidden_size_text_rnn,
                vocabulary_size,
                no_answers,
                device):
        super(VQA, self).__init__()
        self.input_size_text_rnn = input_size_text_rnn
        self.hidden_size_text_rnn = hidden_size_text_rnn
        self.no_answers = no_answers
        self.device = device
        
        self.text_embedding = nn.Embedding(vocabulary_size, input_size_text_rnn)
        self.text_rnn = LSTM(input_size=input_size_text_rnn, 
                            hidden_size=hidden_size_text_rnn, 
                            device=device)

        # forget gate weights
        self.linear = nn.Linear(hidden_size_text_rnn, no_answers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_text, x_image=None):
        
        x = self.text_embedding(x_text)
        x = self.text_rnn(x) # batch_size, no_answers
        x = torch.mean(x, dim=1).to(self.device)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x

        
        




