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
        

