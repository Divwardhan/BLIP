import torch.nn as nn


class PatchEmbedding(nn.Module):
    # constructor
    def __init__(self, img_size, patch_size, dim):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            3, dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x