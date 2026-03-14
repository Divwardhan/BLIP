import torch.nn as nn
import torch

from .text_block import TextBlock
from .text_embeddings import TextEmbeddings

class TextEncoder(nn.Module):

    def __init__(self , vocab_size = 30522 , embed_dim = 768 , depth = 12 , heads=12 , max_len = 40):
        super().__init__()

        self.embeddings = TextEmbeddings(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=40
        )

        self.blocks = nn.ModuleList(
            [
                TextBlock(dim=embed_dim , heads= heads) for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self , input_ids):
        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        text_cls = x[:,0]

        return text_cls , x

