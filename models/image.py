from torch import nn
from ViT import ViT

class Image(nn.Module):

    def __init__(self,             
                 no_in_features,
                 no_out_features,
                 no_patches,
                 no_transformer_blocks,
                 no_heads,
                 dropout, 
                 model_type):
        super(Image, self).__init__()

        if model_type == "VIT":
            self.model = ViT(no_in_features=no_in_features,
                         no_out_features=no_out_features,
                         no_patches=no_patches,
                         no_transformer_blocks=no_transformer_blocks,
                         no_heads=no_heads,
                         dropout=dropout)

    def forward(self, x):
        return self.model(x)