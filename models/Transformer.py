import torch
from torch import nn
import torch.nn.functional as F


class Multihead_Attention(nn.Module):

    def __init__(self, no_features, no_heads=8, dropout=0.1):
        super(Multihead_Attention, self).__init__()

        self.no_heads = no_heads
        self.dropout = dropout

        self.Q = nn.Linear(no_features, no_features)
        self.V = nn.Linear(no_features, no_features)
        self.K = nn.Linear(no_features, no_features)

        self.dropout = nn.Dropout(self.dropout)
        self.softmax = nn.Softmax()
        self.layer_norm = nn.LayerNorm(no_features)

        self.mlp = nn.Linear(no_features , no_features)


    def forward(self, x):

        batch_size, no_patches, no_features = x.shape

        x = self.layer_norm(x)

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        no_features_head = no_features // self.no_heads

        q = q.reshape(batch_size, no_patches, self.no_heads, no_features_head).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, no_patches, self.no_heads, no_features_head).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, no_patches, self.no_heads, no_features_head).permute(0, 2, 1, 3)

        attention_score = torch.matmul(q, k.transpose(-1, -2))

        scale = no_features_head ** -0.5
        attention_score = attention_score * scale
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)

        out = torch.matmul(attention_score, v)

        out = out.permute(0, 2, 1, 3).flatten(2)
        out = self.mlp(out)

        return out
    
class FeedForward(nn.Module):
    
    def __init__(self, no_features, dropout=0.):
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(no_features),
            nn.Linear(no_features, no_features*4), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(no_features*4, no_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Transformer(nn.Module):

    def __init__(self, no_features, no_blocks, no_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.no_feature = no_features
        self.no_layers = no_blocks
        self.no_heads = no_heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(no_features)
        self.layers = nn.ModuleList([])

        for _ in range(no_blocks):
            self.layers.append(nn.ModuleList([
                Multihead_Attention(no_features=no_features, no_heads=no_heads, dropout=dropout), 
                FeedForward(no_features, dropout=dropout)
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)