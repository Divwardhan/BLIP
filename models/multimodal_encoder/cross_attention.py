import torch 
import torch.nn as nn
from ..common.attention import MultiHeadAttention

class CrossAttention(nn.Module):

    def __init__(self , dim=768 ,heads=12):
        super().__init__()

        self.attn = MultiHeadAttention(
            input_vec_dim=dim,
            output_vec_dim=dim,
            num_heads=heads
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self , text_tokens , image_tokens):
        q = self.norm(text_tokens)

        k = image_tokens

        v = image_tokens

        attn_out = self.attn(q,k,v)

        return attn_out
