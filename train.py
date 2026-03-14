import torch
from models.text_encoder.text_encoder import TextEncoder

model = TextEncoder()

tokens = torch.randint(0,30522,(2,20))

cls, tok = model(tokens)

print(cls.shape)
print(tok.shape)