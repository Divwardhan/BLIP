import torch 
import torch.nn as nn
from ..common.attention import MultiHeadAttention
from ..common.feed_forward import FeedForward
from ..common.transformer_block import TransformerBlock

class ViTBlock(nn.Module):
    def __init__(self ,dim ,heads , mlp_ratio):

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn = MultiHeadAttention(
            input_vec_dim=dim,
            output_vec_dim=dim,
            num_heads=heads
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim*mlp_ratio)

        self.mlp =FeedForward(dim , hidden_dim)

    def forward(self,x):
        x_attn, _ = self.attn(self.norm1(x))
        x =x+ x_attn
        x = x+self.mlp(self.norm2(x))
        return x

