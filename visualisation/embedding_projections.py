import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embedding_projection(image_emb, text_emb):

    emb = torch.cat([image_emb, text_emb], dim=0)

    emb = emb.detach().cpu()

    tsne = TSNE(n_components=2)

    proj = tsne.fit_transform(emb)

    N = image_emb.shape[0]

    plt.figure(figsize=(6,6))

    plt.scatter(proj[:N,0], proj[:N,1], label="Images")
    plt.scatter(proj[N:,0], proj[N:,1], label="Text")

    plt.legend()

    plt.title("Image-Text Embedding Space")

    plt.show()