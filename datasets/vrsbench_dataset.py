import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


class VRSBenchDataset(Dataset):
    def __init__(self, root_dir, split="train", max_length=20):
        self.root_dir = root_dir
        self.max_length = max_length

        if split not in {"train", "val"}:
            raise ValueError("split must be one of {'train', 'val'}")

        if split == "train":
            self.img_dir = os.path.join(root_dir, "Images_train/Images_train")
            self.ann_dir = os.path.join(root_dir, "Annotations_train/Annotations_train")
        else:
            self.img_dir = os.path.join(root_dir, "Images_val/Images_val")
            self.ann_dir = os.path.join(root_dir, "Annotations_val/Annotations_val")

        self.files = sorted([f for f in os.listdir(self.ann_dir) if f.endswith(".json")])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ann_path = os.path.join(self.ann_dir, self.files[idx])

        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)

        caption = data["caption"]
        img_path = os.path.join(self.img_dir, data["image"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return image, tokens
