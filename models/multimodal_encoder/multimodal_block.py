import torch 
import torch.nn as nn
from .cross_attention import CrossAttention
from ..common.attention import MultiHeadAttention
from ..common.feed_forward import FeedForward

class MultiModalBlock(nn.Module):
    def __init__(self , dim=768 , heads=12 , mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.self_attn = MultiHeadAttention(
            input_vec_dim=dim,
            output_vec_dim=dim,
            num_heads=heads
        )

        self.cross_attn= CrossAttention(
            dim=dim,
            heads=heads
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.ff = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim
        )
    
    def forward(self , text_tokens , image_tokens):
        q =self.norm1(text_tokens)

        self_attn = self.self_attn(text_tokens)

        text_tokens= text_tokens+self_attn

        cross_attn = self.cross_attn(text_tokens , image_tokens)

        text_tokens= text_tokens+cross_attn

        text_tokens = text_tokens+self.ff(self.norm2(text_tokens))

        return text_tokens
