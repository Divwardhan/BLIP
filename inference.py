import argparse

import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms

from models.blip_model import BLIP


def load_image(image_path, image_size=224):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


@torch.no_grad()
def generate_caption(model, tokenizer, image_tensor, max_len=20, device="cpu"):
    model.eval()

    image_tensor = image_tensor.to(device)

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    generated = torch.full((1, 1), cls_token_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        outputs = model(image_tensor, generated)
        next_token_logits = outputs["caption_logits"][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == sep_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="BLIP caption inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BLIP(vocab_size=tokenizer.vocab_size).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)

    image = load_image(args.image)
    caption = generate_caption(model, tokenizer, image, max_len=args.max_len, device=args.device)

    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
