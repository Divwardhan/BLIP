import torch 
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_vec_dim , output_vec_dim ):
        super().__init__()
        torch.manual_seed(0)

        self.wq = nn.Linear(input_vec_dim, output_vec_dim , bias = False)
        self.wk = nn.Linear(input_vec_dim , output_vec_dim , bias = False)
        self.wv = nn.Linear(input_vec_dim , output_vec_dim , bias = False)

    def forward(self , q ,k = None , v=None):
        if k is None:
            k=q
        if v is None:
            v=q

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        att_score = (q @ torch.transpose(k, -2,-1)) / k.shape[-1]**0.5

        attn_score_softmax = F.softmax(att_score, dim = -1)

        weighted_values = attn_score_softmax @ v

        return weighted_values

class MultiHeadAttention(nn.Module):
    def __init__(self , input_vec_dim , output_vec_dim , num_heads):
        super().__init__()

        assert output_vec_dim%num_heads==0

        self.num_heads = num_heads

        self.head_dim = output_vec_dim//num_heads

        self.heads = nn.ModuleList(
            [SelfAttention(input_vec_dim , self.head_dim) for _ in range(num_heads)]
        )

        self.out_proj = nn.Linear(output_vec_dim , output_vec_dim)

    def forward(self , q,k=None,v= None):
        head_outputs = [head(q,k,v) for head in self.heads]

        concat_heads = torch.cat(head_outputs , dim=-1)

        output = self.out_proj(concat_heads)

        return output
    
x = torch.randn(1, 5, 4)  
# batch = 2
# sequence length = 5
# embedding dimension = 16
print("Inout X")
print(x)

mha = MultiHeadAttention(input_vec_dim=4, output_vec_dim=4, num_heads=4)

out = mha(x)

print("Output")
print(out)