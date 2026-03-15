import torch.nn.functional as F


def caption_loss(logits, targets, ignore_index=0):
    bsz, seq_len, vocab = logits.shape

    logits = logits.reshape(bsz * seq_len, vocab)
    targets = targets.reshape(bsz * seq_len)

    return F.cross_entropy(logits, targets, ignore_index=ignore_index)
