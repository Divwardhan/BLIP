import matplotlib.pyplot as plt


def plot_training_curves(itc_losses, itm_losses, lm_losses):

    plt.figure(figsize=(8,5))

    plt.plot(itc_losses, label="ITC Loss")
    plt.plot(itm_losses, label="ITM Loss")
    plt.plot(lm_losses, label="LM Loss")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")

    plt.title("BLIP Training Losses")

    plt.legend()

    plt.tight_layout()
    plt.show()