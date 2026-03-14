import torch.nn as nn
import torch 
from ..common.attention import MultiHeadAttention
from ..common.feed_forward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self , dim=768 , heads = 12 , mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.self_attn = MultiHeadAttention(
            input_vec_dim=dim,
            output_vec_dim=dim,
            num_heads=heads
        )

        self.norm2 = nn.LayerNorm(dim)

        self.cross_attn = MultiHeadAttention(
            input_vec_dim=dim,
            output_vec_dim=dim,
            num_heads=heads
        )

        self.norm3 = nn.LayerNorm(dim)

        hidden_dim = int(dim*mlp_ratio)

        self.ff = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim
        )

    def forward(self , text_tokens , image_tokens , causal_mask=None):

        q = self.norm1(text_tokens)

        self_out = self.self_attn(q)

        text_tokens= text_tokens+self_out

        q=self.norm2(text_tokens)

        cross_out = self.cross_attn(
            q,
            image_tokens,
            image_tokens
        )
        text_tokens = text_tokens+cross_out

        text_tokens= text_tokens+self.ff(self.norm3(text_tokens))

        return text_tokens