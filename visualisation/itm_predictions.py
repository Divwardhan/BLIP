import torch
import matplotlib.pyplot as plt


def plot_itm_predictions(logits):

    probs = torch.softmax(logits, dim=-1)

    match_probs = probs[:,1].detach().cpu()

    plt.figure(figsize=(6,4))

    plt.bar(range(len(match_probs)), match_probs)

    plt.ylim(0,1)

    plt.title("ITM Match Probabilities")
    plt.xlabel("Sample")
    plt.ylabel("P(match)")

    plt.tight_layout()
    plt.show()