import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.vrsbench_dataset import VRSBenchDataset
from losses.caption_loss import caption_loss
from losses.contrastive_loss import contrastive_loss
from losses.itm_loss import itm_loss
from models.blip_model import BLIP
from visualisation.itm_predictions import plot_itm_predictions
from visualisation.similarity_matrix import plot_similarity_matrix


plt.ion()


def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP on VRSBench")
    parser.add_argument("--data-root", type=str, required=True, help="VRSBench root directory")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--save-path", type=str, default="blip_vrsbench.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BLIP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = VRSBenchDataset(root_dir=args.data_root, split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    itc_losses, itm_losses, lm_losses = [], [], []

    for epoch in range(args.epochs):
        model.train()

        for step, (images, tokens) in enumerate(dataloader):
            images = images.to(device)
            tokens = tokens.to(device)
            batch_size = images.shape[0]

            decoder_input = tokens[:, :-1]
            caption_targets = tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images, decoder_input)
                itc_logits = outputs["contrastive_logits"]
                caption_logits = outputs["caption_logits"]

                loss_itc = contrastive_loss(itc_logits)
                loss_lm = caption_loss(caption_logits, caption_targets)

            perm = torch.randperm(batch_size, device=device)
            negative_tokens = tokens[perm]

            itm_images = torch.cat([images, images], dim=0)
            itm_tokens = torch.cat([tokens, negative_tokens], dim=0)
            itm_labels = torch.cat(
                [
                    torch.ones(batch_size, device=device),
                    torch.zeros(batch_size, device=device),
                ]
            ).long()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                itm_outputs = model(itm_images, itm_tokens[:, :-1])
                itm_logits = itm_outputs["itm_logits"]
                loss_itm = itm_loss(itm_logits, itm_labels)

            loss = loss_itc + loss_itm + loss_lm

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

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

            if step % 50 == 0:
                plot_similarity_matrix(itc_logits)
                plot_itm_predictions(itm_logits)

                plt.show()
                plt.pause(0.001)
                plt.close("all")

    torch.save(model.state_dict(), args.save_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(itc_losses, label="ITC Loss")
    ax.plot(itm_losses, label="ITM Loss")
    ax.plot(lm_losses, label="Caption Loss")

    ax.legend()
    ax.set_title("Training Curves")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
