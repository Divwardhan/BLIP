import torch 
import torch.nn as nn

from .multimodal_block import MultiModalBlock

class MultiModalEncoder(nn.Module):
    def __init__(self, depth =6 ,dim=768 , heads = 12 ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MultiModalBlock(
                dim=dim,
                heads=heads
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self , text_tokens , image_tokens):
        x = text_tokens

        for block in self.blocks:
            x = block(x,image_tokens)
        
        x=self.norm(x)

        multimodal_cls = x[:,0]

        return multimodal_cls,x