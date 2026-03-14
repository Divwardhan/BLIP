import matplotlib.pyplot as plt
import seaborn as sns


def plot_similarity_matrix(sim_matrix):

    sim = sim_matrix.detach().cpu()

    plt.figure(figsize=(6,6))

    sns.heatmap(sim, cmap="viridis")

    plt.title("Image-Text Similarity Matrix")

    plt.xlabel("Text")
    plt.ylabel("Image")

    plt.show()

