import torch
from torch import nn
import torch.nn.functional as F
from graphpfn.base_model import GraphPFNModel
from tfmplayground.model import TransformerEncoderStack
from tfmplayground.utils import get_default_device


class BinaryGCNModel(GraphPFNModel):
    
    def _make_transformer_encoder(self) -> nn.Module:
        
        class TransformerWrapper(nn.Module):
            def __init__(self, transformer: nn.Module, gcn: nn.Module):
                super().__init__()
                self.transformer = transformer
                self.gcn = gcn

            def forward(self, x: torch.Tensor, single_eval_pos: int, **kwargs) -> torch.Tensor:
                adj = kwargs['adj']
                graph_embeddings = self.gcn(adj)
                x += graph_embeddings.unsqueeze(0).unsqueeze(0)
                return self.transformer(x, single_eval_pos)
        
        encoder = TransformerEncoderStack(
            self.num_layers,
            self.embedding_size,
            self.num_attention_heads,
            self.mlp_hidden_size,
        )
        graph_encoder = GCN(self.embedding_size, hidden_dim=self.embedding_size)
        
        return TransformerWrapper(encoder, graph_encoder)
    

class GCN(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Features: (in-degree, out-degree)
        self.input_proj = nn.Linear(2, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def _get_structural_features(self, adj: torch.Tensor) -> torch.Tensor:
        out_degree = adj.sum(dim=1, keepdim=True) 
        in_degree = adj.sum(dim=0, keepdim=True).t()
        return torch.cat([in_degree, out_degree], dim=1)

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        N = adj.shape[0]
        # Add self-loops and row-normalize
        adj_hat = adj + torch.eye(N, device=adj.device)
        row_sum = adj_hat.sum(1)
        d_inv = torch.pow(row_sum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        return torch.diag(d_inv) @ adj_hat

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.to(get_default_device())
        
        # 1. Structural features (N, 2)
        x = self._get_structural_features(adj)
        
        # 2. Adjacency normalization
        norm_adj = self._normalize_adj(adj)
        
        # 3. Message Passing
        x = F.relu(self.input_proj(x))
        
        # Neighborhood Aggregation (Matrix Mult) -> Linear Layer -> Activation
        x = norm_adj @ x
        x = F.relu(self.layer1(x))
        
        x = norm_adj @ x
        return self.layer2(x)