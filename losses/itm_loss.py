import torch.nn.functional as F

def itm_loss(logits, labels):

    return F.cross_entropy(logits, labels)