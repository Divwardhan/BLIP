import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveHead(nn.Module):

    def __init__(self , dim=768 , temperature = 0.07):
        super().__init__()

        self.image_proj = nn.Linear(dim ,dim)
        self.text_proj = nn.Linear(dim , dim)

        self.temperature = temperature

    def forward(self , image_cls , text_cls):
        image_emb =F.normalize(self.image_proj(image_cls) , dim=-1)
        text_emb = F.normalize(self.text_proj(text_cls) , dim=-1)

        logits = image_emb@text_emb.T/self.temperature

        return logits
    

    