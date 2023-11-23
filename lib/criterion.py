import torch 
from torch import nn


class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

    def forward(self, scores, targets):

        batch_size, no_classes = scores.shape
        
        L1_dist = torch.abs(scores - targets).sum() // batch_size

        return L1_dist