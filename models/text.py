from torch import nn
from LSTM import LSTM, LSTMDeep

class Text(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 device, 
                 model_type):
        super(Text, self).__init__()

        if model_type == "LSTM":
            self.model = LSTM(input_size=input_size, 
                              hidden_size=hidden_size, 
                              device=device)
        if model_type == "LSTMDeep":
            self.model = LSTMDeep(input_size=input_size, 
                                  hidden_size=hidden_size, 
                                  output_size= 1024,
                                  device=device)

    def forward(self, x):
        return self.model(x)
