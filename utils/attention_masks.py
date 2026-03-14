import torch


def causal_mask(seq_len):

    mask = torch.triu(
        torch.ones(seq_len, seq_len),
        diagonal=1
    ).bool()

    return mask