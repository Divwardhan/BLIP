import torch.nn as nn
import torch
from ..common.attention import MultiHeadAttention
from ..common.feed_forward import FeedForward

class TextBlock(nn.Module):
    def __init__(self , dim = 768 , heads =12 , mlp_ratio = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.attn = MultiHeadAttention(
            input_vec_dim=dim,
            output_vec_dim=dim,
            num_heads=heads
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim*mlp_ratio)

        self.mlp = FeedForward(
            dim = dim,
            hidden_dim= hidden_dim
        )
    
    def forward(self ,x):
        attn_out =self.attn(self.norm1(x))

        x=x+attn_out

        x=x+self.mlp(self.norm2(x))

        return x
    