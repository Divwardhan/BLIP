import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.blip_model import BLIP

from losses.contrastive_loss import contrastive_loss
from losses.itm_loss import itm_loss
from losses.caption_loss import caption_loss

from visualisation.similarity_matrix import plot_similarity_matrix
from visualisation.itm_predictions import plot_itm_predictions
from visualisation.training_curves import plot_training_curves

# IMPORT YOUR DATASET
from datasets.vrsbench_dataset import VRSBenchDataset


# enable inline plotting
plt.ion()


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# model
model = BLIP().to(device)


# optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)


# mixed precision
scaler = torch.cuda.amp.GradScaler()


########################################
# VRSBench Dataset
########################################

dataset = VRSBenchDataset(
    root_dir="/kaggle/input/datasets/divwardhanagrawal/vrs-bench",
    split="train"
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# logging
itc_losses = []
itm_losses = []
lm_losses = []


# training params
epochs = 3


for epoch in range(epochs):

    model.train()

    for step, (images, tokens) in enumerate(dataloader):

        images = images.to(device)
        tokens = tokens.to(device)

        # teacher forcing shift
        decoder_input = tokens[:, :-1]
        caption_targets = tokens[:, 1:]

        # ITM labels (dummy)
        batch_size = images.shape[0]
        itm_labels = torch.ones(batch_size).long().to(device)

        optimizer.zero_grad()


        # mixed precision forward
        with torch.amp.autocast(device_type=device.type):

            outputs = model(images, decoder_input)

            itc_logits = outputs["contrastive_logits"]
            itm_logits = outputs["itm_logits"]
            caption_logits = outputs["caption_logits"]

            loss_itc = contrastive_loss(itc_logits)
            loss_itm = itm_loss(itm_logits, itm_labels)
            loss_lm = caption_loss(caption_logits, caption_targets)

            loss = loss_itc + loss_itm + loss_lm


        # backward
        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()


        # logging
        itc_losses.append(loss_itc.item())
        itm_losses.append(loss_itm.item())
        lm_losses.append(loss_lm.item())


        if step % 20 == 0:

            print(
                f"Epoch {epoch} Step {step} | "
                f"ITC {loss_itc:.3f} | "
                f"ITM {loss_itm:.3f} | "
                f"LM {loss_lm:.3f}"
            )


        # visualization
        if step % 50 == 0:

            plot_similarity_matrix(itc_logits)
            plot_itm_predictions(itm_logits)

            plt.show()
            plt.pause(0.001)
            plt.close('all')


fig, ax = plt.subplots(figsize=(8,5))

ax.plot(itc_losses, label="ITC Loss")
ax.plot(itm_losses, label="ITM Loss")
ax.plot(lm_losses, label="Caption Loss")

ax.legend()
ax.set_title("Training Curves")

plt.tight_layout()
plt.show()

plt.show()