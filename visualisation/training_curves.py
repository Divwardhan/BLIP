import matplotlib.pyplot as plt

def plot_training_curves(itc, itm, lm):

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(itc, label="ITC Loss")
    ax.plot(itm, label="ITM Loss")
    ax.plot(lm, label="Caption Loss")

    ax.legend()
    ax.set_title("Training Curves")

    plt.tight_layout()
    plt.show()   