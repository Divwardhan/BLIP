import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):

    def __init__(self , dim , heads , mlp_dim):
        super().__init__()

        self.attn = MultiHeadAttention(dim , dim , heads)
        self.ff = FeedForward(dim , mlp_dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self , x):
        x = x+self.attn(self.norm1(x))
        x = x+self.ff(self.norm2(x))

        return x 
    
