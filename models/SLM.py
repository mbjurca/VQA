import torch
import torch.nn as nn
from Transformer import Transformer

# Small language model
class SLM(nn.Module):

    def __init__(self, no_features, sequence_length, no_transformer_blocks=8, no_heads=3, dropout=0.1):
        super(SLM, self).__init__()
        self.positional_encodding = nn.Parameter(torch.randn(1, sequence_length, no_features))
        self.transformer = Transformer(no_features=no_features, no_blocks=no_transformer_blocks, no_heads=no_heads, dropout=dropout)
    
    def forward(self, x):

        x = x + self.positional_encodding
        x = self.transformer(x)
        x = x.mean(1)

        return x