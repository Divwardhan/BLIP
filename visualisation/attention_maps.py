import matplotlib.pyplot as plt
import torch


def visualize_attention(attn_weights, image):

    attn = attn_weights.mean(0).detach().cpu()

    attn = attn.reshape(14,14)

    plt.figure(figsize=(6,6))

    plt.imshow(image)
    plt.imshow(attn, cmap="jet", alpha=0.5)

    plt.title("Cross-Attention Map")

    plt.axis("off")

    plt.show()