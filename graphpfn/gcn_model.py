from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear
from graphpfn.base_model import GraphPFNModel
from tfmplayground.utils import get_default_device


@dataclass(eq=False)
class GCNModel(GraphPFNModel):
    
    def _make_transformer_encoder(self) -> nn.Module:
        
        class TransformerWrapper(nn.Module):
            def __init__(self, transformer: nn.Module, gcn: nn.Module):
                super().__init__()
                self.transformer = transformer
                self.gcn = gcn

            def forward(self, x: torch.Tensor, single_eval_pos: int, **kwargs) -> torch.Tensor:
                prob_adj = kwargs['prob_adj']
                graph_embeddings = self.gcn(prob_adj)
                return self.transformer(x, single_eval_pos, graph_embed=graph_embeddings)
        
        encoder = TransformerEncoderStack(
            self.num_layers,
            self.embedding_size,
            self.num_attention_heads,
            self.mlp_hidden_size,
        )
        graph_encoder = GCN(self.embedding_size, hidden_dim=self.embedding_size)
        
        return TransformerWrapper(encoder, graph_encoder)


class TransformerEncoderStack(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 embedding_size: int, 
                 num_attention_heads: int,
                 mlp_hidden_size: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, 
                                                                   num_attention_heads,
                                                                   mlp_hidden_size))

    def forward(self, x: torch.Tensor, single_eval_position: int, graph_embed: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position, graph_embed=graph_embed)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int, batch_first: bool = True):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first)
        self.self_attention_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first)

        self.linear1 = Linear(embedding_size, mlp_hidden_size)
        self.linear2 = Linear(mlp_hidden_size, embedding_size)

        self.norm1 = AdaLN(embedding_size)
        self.norm2 = AdaLN(embedding_size)
        self.norm3 = AdaLN(embedding_size)

    def forward(self, src: torch.Tensor, single_eval_position: int, graph_embed: torch.Tensor) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        # adjacency based attention
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src, graph_embed)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attention_between_datapoints(src[:,:single_eval_position], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        # test data attends to the training data
        src_right = self.self_attention_between_datapoints(src[:,single_eval_position:], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src, graph_embed)
        # MLP after attention
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src, graph_embed)
        return src
    

class GCN(nn.Module):
    
    def __init__(self, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # one embedding for features, one for the target
        self.node_type_embed = nn.Embedding(2, hidden_dim)
        
        # linear layers for forward and reverse message passing
        self.fwd_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.rev_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fwd_layer2 = nn.Linear(hidden_dim, out_dim)
        self.rev_layer2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def _get_positional_encoding(self, n: int, d: int, device) -> torch.Tensor:
        pe = torch.zeros(n, d, device=device)
        position = torch.arange(0, n, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2, device=device).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _normalize_directed_adj(self, adj: torch.Tensor) -> torch.Tensor:
        # add self-loops to ensure nodes retain their own features
        adj_hat = adj + torch.eye(adj.shape[0], device=adj.device)
        row_sum = adj_hat.sum(dim=1, keepdim=True)
        # avoid division by zero
        mask = row_sum == 0
        row_sum[mask] = 1.0
        return adj_hat / row_sum

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        device = get_default_device()
        adj = adj.to(device)
        N = adj.shape[0]
        
        # initial embedding
        node_indices = torch.zeros(N, dtype=torch.long, device=device)
        node_indices[-1] = 1
        x = self.node_type_embed(node_indices) + self._get_positional_encoding(N, self.hidden_dim, device)
        
        # normalized adjacency matrices
        norm_adj_fwd = self._normalize_directed_adj(adj)
        norm_adj_rev = self._normalize_directed_adj(adj.t())
        
        out_fwd = self.fwd_layer1(norm_adj_fwd @ x)
        out_rev = self.rev_layer1(norm_adj_rev @ x)
        x = F.relu(self.ln1(out_fwd + out_rev))
        out_fwd = self.fwd_layer2(norm_adj_fwd @ x)
        out_rev = self.rev_layer2(norm_adj_rev @ x)
        x = self.ln2(out_fwd + out_rev)
        
        return x
    

class AdaLN(nn.Module):
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embedding_dim)
        self.scale = nn.Linear(embedding_dim, embedding_dim)
        self.shift = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln(x) # (B, R, C, D)
        scale = 1.0 + self.scale(graph_emb).unsqueeze(0).unsqueeze(0) # (C, D) -> (1, 1, C, D)
        shift = self.shift(graph_emb).unsqueeze(0).unsqueeze(0) # (C, D) -> (1, 1, C, D)
        
        return scale * x_norm + shift