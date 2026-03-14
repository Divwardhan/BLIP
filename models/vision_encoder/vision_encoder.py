import torch
import torch.nn as nn

from .patch_embedding import PatchEmbedding
from .vit_block import ViTBlock


class VisionEncoder(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        heads=12
    ):

        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        B = x.shape[0]

        x = self.patch_embed(x)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # positional embeddings
        x = x + self.pos_embed[:, :x.size(1)]

        # transformer layers
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        image_cls = x[:, 0]
        image_tokens = x[:, 1:]

        return image_cls, image_tokens
