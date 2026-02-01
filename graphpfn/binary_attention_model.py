import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, LayerNorm

from graphpfn.base_model import GraphPFNModel
from tfmplayground.utils import get_default_device


class BinaryAttentionModel(GraphPFNModel):
    def __init__(self,
                 embedding_size: int,
                 num_attention_heads: int,
                 num_graph_attention_heads: int,
                 mlp_hidden_size: int,
                 num_layers: int,
                 num_outputs: int):
        super().__init__(embedding_size,
                         num_attention_heads,
                         mlp_hidden_size,
                         num_layers,
                         num_outputs)
        self.num_graph_attention_heads = num_graph_attention_heads
        self.architecture['num_graph_attention_heads'] = num_graph_attention_heads

    def _make_transformer_encoder(self) -> nn.Module:
        encoder = TransformerEncoderStack(
            self.num_layers,
            self.embedding_size,
            self.num_attention_heads,
            self.num_graph_attention_heads,
            self.mlp_hidden_size,
        )
        return encoder
    

class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, num_attention_heads: int, num_graph_attention_heads: int, mlp_hidden_size: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, num_graph_attention_heads, mlp_hidden_size))

    def forward(self, x: torch.Tensor, single_eval_position: int, **kwargs) -> torch.Tensor:
        adj = kwargs['adj']
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position, adj=adj)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, nhead_graph: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True):
        super().__init__()
        self.self_attn_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first)
        self.self_attn_graph = MultiheadAttention(embedding_size, 2 * nhead_graph, batch_first=batch_first)
        self.nhead_graph = nhead_graph

        self.linear1 = Linear(embedding_size, mlp_hidden_size)
        self.linear2 = Linear(mlp_hidden_size, embedding_size)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps)

    def forward(self, src: torch.Tensor, single_eval_position: int, adj: torch.Tensor) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        # adjacency based attention
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        # flip adjacency matrix, except for diagonal entries
        eye = torch.eye(col_size, dtype=torch.bool).bool()
        adj = adj.bool()
        mask_1 = ~((adj | eye).to(get_default_device()))
        mask_2 = ~((adj.T | eye).to(get_default_device()))
        mask_1 = mask_1.unsqueeze(0).expand(self.nhead_graph, -1, -1)
        mask_2 = mask_2.unsqueeze(0).expand(self.nhead_graph, -1, -1)
        # (2 * nhead_graph, C, C)
        mask = torch.cat([mask_1, mask_2], dim=0)
        # (B * R, 2 * nhead_graph, C, C)
        mask = mask.unsqueeze(0).repeat(batch_size * rows_size, 1, 1, 1)
        # (B * R * 2 * nhead_graph, C, C) - required shape for MultiheadAttention
        mask = mask.view(batch_size * rows_size * 2 * self.nhead_graph, col_size, col_size)
        
        src = self.self_attn_graph(src, src, src, attn_mask=mask)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attn_between_datapoints(src[:,:single_eval_position], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        # test data attends to the training data
        src_right = self.self_attn_between_datapoints(src[:,single_eval_position:], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src