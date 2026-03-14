import torch
import torch.nn as nn

from .decoder_block import DecoderBlock
from utils.attention_masks import causal_mask


class TextDecoder(nn.Module):

    def __init__(
        self,
        vocab_size=30522,
        dim=768,
        depth=6,
        heads=12,
        max_len=40
    ):

        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        self.blocks = nn.ModuleList([
            DecoderBlock(dim, heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        self.lm_head = nn.Linear(dim, vocab_size)

        self.max_len = max_len

    def forward(self, input_ids, image_tokens):

        B, L = input_ids.shape

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_embed(input_ids) + self.pos_embed(positions)

        mask = causal_mask(L).to(input_ids.device)

        for block in self.blocks:
            x = block(x, image_tokens, mask)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits