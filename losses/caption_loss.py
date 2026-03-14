import torch.nn.functional as F

def caption_loss(logits, targets):

    B, L, V = logits.shape

    logits = logits.reshape(B * L, V)
    targets = targets.reshape(B * L)

    loss = F.cross_entropy(logits, targets)

    return loss
