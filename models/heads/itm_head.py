import torch 
import torch.nn as nn

class ITMHead(nn.Module):
    def __init__(self , dim=768):
        super().__init__()
        self.classifier = nn.Linear(dim,2)

    def forward(self , multimodal_cls):
        logits = self.classifier(multimodal_cls)
        return logits
    