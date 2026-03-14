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
    batch_size=2,
    shuffle=True
)


# logging
itc_losses = []
itm_losses = []
lm_losses = []


# training params
epochs = 7

for epoch in range(epochs):

    model.train()

    for step, (images, tokens) in enumerate(dataloader):

        images = images.to(device)
        tokens = tokens.to(device)

        batch_size = images.shape[0]

        # caption teacher forcing
        decoder_input = tokens[:, :-1]
        caption_targets = tokens[:, 1:]

        optimizer.zero_grad()

        ################################
        # CONTRASTIVE + CAPTION FORWARD
        ################################
        with torch.amp.autocast(device_type=device.type):

            outputs = model(images, decoder_input)

            itc_logits = outputs["contrastive_logits"]
            caption_logits = outputs["caption_logits"]

            loss_itc = contrastive_loss(itc_logits)
            loss_lm = caption_loss(caption_logits, caption_targets)

        ################################
        # ITM NEGATIVE SAMPLING
        ################################

        perm = torch.randperm(batch_size)

        negative_tokens = tokens[perm]

        itm_images = torch.cat([images, images], dim=0)
        itm_tokens = torch.cat([tokens, negative_tokens], dim=0)

        itm_labels = torch.cat([
            torch.ones(batch_size),
            torch.zeros(batch_size)
        ]).long().to(device)

        ################################
        # ITM FORWARD
        ################################

        with torch.amp.autocast(device_type=device.type):

            itm_outputs = model(itm_images, itm_tokens[:, :-1])

            itm_logits = itm_outputs["itm_logits"]

            loss_itm = itm_loss(itm_logits, itm_labels)

        ################################
        # TOTAL LOSS
        ################################

        loss = loss_itc + loss_itm + loss_lm

        ################################
        # BACKPROP
        ################################

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        ################################
        # LOGGING
        ################################

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

        ################################
        # VISUALIZATION
        ################################

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