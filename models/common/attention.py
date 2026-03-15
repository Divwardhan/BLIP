import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, input_vec_dim, output_vec_dim, num_heads):
        super().__init__()

        if output_vec_dim % num_heads != 0:
            raise ValueError("output_vec_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = output_vec_dim // num_heads

        self.q_proj = nn.Linear(input_vec_dim, output_vec_dim, bias=False)
        self.k_proj = nn.Linear(input_vec_dim, output_vec_dim, bias=False)
        self.v_proj = nn.Linear(input_vec_dim, output_vec_dim, bias=False)
        self.out_proj = nn.Linear(output_vec_dim, output_vec_dim)

    def _split_heads(self, x):
        batch, seq_len, dim = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, q, k=None, v=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        q = self._split_heads(self.q_proj(q))
        k = self._split_heads(self.k_proj(k))
        v = self._split_heads(self.v_proj(v))

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = attn_weights @ v

        context = context.transpose(1, 2).contiguous()
        batch, seq_len, _, _ = context.shape
        context = context.view(batch, seq_len, self.num_heads * self.head_dim)

        return self.out_proj(context)
