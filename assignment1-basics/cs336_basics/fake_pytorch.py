"""Pedagogical Implementation of Some Pytorch Modules for Use in Transformer"""

import torch
from einops import rearrange, einsum
from jaxtyping import Float
import math

def init_linear_weights(in_features, out_features):
    W = torch.nn.Parameter(torch.zeros([out_features, in_features]))
    std_dev = math.sqrt(2 / (in_features + out_features))
    torch.nn.init.trunc_normal_(W, mean=0.0, std=std_dev, a=-3.0*std_dev, b=3.0*std_dev)
    return W

def batched_matmul(W, x):
    return einsum(W, x, "d_out d_in, ... d_in -> ... d_out")

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.W = init_linear_weights(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return batched_matmul(self.W, x)
        

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

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = init_linear_weights(in_features=d_model, out_features=d_ff)
        self.W2 = init_linear_weights(in_features=d_ff, out_features=d_model)
        self.W3 = init_linear_weights(in_features=d_model, out_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = lambda z : z * torch.sigmoid(z)
        return batched_matmul(self.W2, silu(batched_matmul(self.W1, x)) * batched_matmul(self.W3, x))
        return batched_matmul(self.W2, i1)
    
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        angle = lambda seq, dim : seq / (theta ** (dim / d_k))
        
        rotations = []
        for i in range(max_seq_len):
            rotations.append([])
            for dim in range(0, d_k, 2):
                angle_ = angle(i, dim)
                rotations[-1].append([[math.cos(angle_), -1 * math.sin(angle_)], 
                                    [math.sin(angle_), math.cos(angle_)]])
        self.register_buffer('rotations_by_position', torch.Tensor(rotations))
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        rotation_matrix = []
        for position in self.rotations_by_position[token_positions]:
            rotation_matrix.append(torch.block_diag(*position))
        return einsum(x, torch.stack(rotation_matrix), "... seq inner_dim, seq outer_dim inner_dim -> ... seq outer_dim")

def softmax(x: torch.Tensor, dim: int):
    """Apply softmax to ith dimension of tensor x."""
    result = x.clone()
    result -= torch.max(result, dim=dim, keepdim=True).values
    result = torch.exp(result) / torch.sum(torch.exp(result), dim=dim, keepdim=True)
    return result
    
def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
    ) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    qtk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    masked_qtk = qtk.masked_fill(~mask, -float('inf')) if mask is not None else qtk
    return einsum(softmax(masked_qtk / math.sqrt(d_k), dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")
