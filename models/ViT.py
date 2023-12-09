import torch
import torch.nn as nn
from Transformer import Transformer

class Feature_Extractor(nn.Module):

    def __init__(self, num_in_features, num_out_features):
        super(Feature_Extractor, self).__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.feat_extractor_layer = nn.Linear(num_in_features, num_out_features)

    def forward(self, x):
        
        batch_size, _, num_features = x.shape
        x = x.reshape(batch_size, -1, self.num_in_features)
        x = self.feat_extractor_layer(x).reshape(batch_size, -1, self.num_out_features)
        
        return x

class ViT(nn.Module):

    def __init__(self, no_in_features, no_out_features, no_patches, no_transformer_blocks=8, no_heads=3, dropout=0.1):
        super(ViT, self).__init__()
        self.feature_extractor = Feature_Extractor(num_in_features=no_in_features, num_out_features=no_out_features)
        self.positional_encodding = nn.Parameter(torch.randn(1, no_patches, no_out_features))
        self.transformer = Transformer(no_features=no_out_features, no_blocks=no_transformer_blocks, no_heads=no_heads, dropout=dropout)
    
    def forward(self, x):

        x = self.feature_extractor(x)
        x = x + self.positional_encodding
        x = self.transformer(x)
        x = x.mean(1)

        return x
