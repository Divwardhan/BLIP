import matplotlib.pyplot as plt
import torch

def plot_itm_predictions(logits):

    probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(probs, bins=20)
    ax.set_title("ITM Match Probabilities")

    plt.tight_layout()
    plt.show()    # REQUIRED