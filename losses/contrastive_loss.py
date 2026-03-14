import torch
import torch.nn.functional as F


def contrastive_loss(logits):

    B = logits.size(0)

    labels = torch.arange(B, device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2