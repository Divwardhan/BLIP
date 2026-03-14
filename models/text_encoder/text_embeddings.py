import torch.nn as nn
import torch

class TextEmbeddings(nn.Module):
    def __init__(self,vocab_size = 30522 , embed_dim =768 , max_len=40):

        super().__init__()

        self.token_embed = nn.Embedding(vocab_size , embed_dim)

        self.pos_embed = nn.Embedding(max_len ,embed_dim)


    def forward(self , input_ids):
        B,L = input_ids.shape

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        token_embeddings = self.token_embed(input_ids)

        pos_embeddings = self.pos_embed(positions)

        return token_embeddings + pos_embeddings