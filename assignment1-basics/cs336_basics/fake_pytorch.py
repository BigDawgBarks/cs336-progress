"""Pedagogical Implementation of Some Pytorch Modules for Use in Transformer"""

import torch
from einops import rearrange, einsum
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros([out_features, in_features]))
        std_dev = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std_dev, a=-3.0*std_dev, b=3.0*std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
        

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.zeros([num_embeddings, embedding_dim]))
        torch.nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.scale = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model + self.eps)
        result = einsum(x, 1/rms, self.scale, "... d_model, ..., d_model -> ... d_model")

        return result.to(in_dtype)