import matplotlib.pyplot as plt

def plot_similarity_matrix(sim):

    sim = sim.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(sim, cmap="viridis")
    fig.colorbar(im)

    ax.set_title("Image-Text Similarity Matrix")

    plt.tight_layout()
    plt.show()     