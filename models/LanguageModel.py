from torch import nn
from SLM import SLM

class LanguageModel(nn.Module):

    def __init__(self,
                 no_features,
                 sequence_length,
                 no_transformer_blocks,
                 no_heads,
                 dropout,
                 model_type):
        
        super(LanguageModel, self).__init__()

        if model_type == "SLM":
            self.model = SLM(no_features,
                             sequence_length,
                             no_transformer_blocks,
                             no_heads,
                             dropout)

    def forward(self, x):
        return self.model(x)