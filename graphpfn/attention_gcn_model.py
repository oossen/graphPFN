from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear

from graphpfn.base_model import GraphPFNModel
from tfmplayground.utils import get_default_device

from graphpfn.attention_model import MultiplicativeMultiheadAttention
from graphpfn.gcn_model import GCN, AdaLN




@dataclass(eq=False)
class AttentionGCNModel(GraphPFNModel):
    num_feature_attention_heads: int
    num_graph_attention_heads: int
    
    def _make_transformer_encoder(self) -> nn.Module:
        
        class TransformerWrapper(nn.Module):
            def __init__(self, transformer: nn.Module, gcn: nn.Module):
                super().__init__()
                self.transformer = transformer
                self.gcn = gcn

            def forward(self, x: torch.Tensor, single_eval_pos: int, **kwargs) -> torch.Tensor:
                prob_adj = kwargs['prob_adj']
                graph_embeddings = self.gcn(prob_adj)
                return self.transformer(x, single_eval_pos, prob_adj, graph_embeddings)
        
        encoder = TransformerEncoderStack(
            self.num_layers,
            self.embedding_size,
            self.num_attention_heads,
            self.num_feature_attention_heads,
            self.num_graph_attention_heads,
            self.mlp_hidden_size,
        )
        graph_encoder = GCN(self.embedding_size, hidden_dim=self.embedding_size)
        
        return TransformerWrapper(encoder, graph_encoder)
    

class TransformerEncoderStack(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 embedding_size: int, 
                 num_attention_heads: int, 
                 num_feature_attention_heads: int,
                 num_graph_attention_heads: int, 
                 mlp_hidden_size: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, 
                                                                   num_attention_heads, 
                                                                   num_feature_attention_heads,
                                                                   num_graph_attention_heads, 
                                                                   mlp_hidden_size))

    def forward(self, x: torch.Tensor, single_eval_position: int, prob_adj, graph_embed) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x, single_eval_position, prob_adj, graph_embed)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, nhead_feature: int, nhead_graph: int, mlp_hidden_size: int, batch_first: bool = True):
        super().__init__()
        self.self_attn_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first)
        # parental heads, child heads, unrestricted heads
        self.n_feature_attn_heads = (nhead_graph // 2, nhead_graph // 2, nhead_feature)
        self.self_attn_graph = MultiplicativeMultiheadAttention(embedding_size, sum(self.n_feature_attn_heads))

        self.linear1 = Linear(embedding_size, mlp_hidden_size)
        self.linear2 = Linear(mlp_hidden_size, embedding_size)

        self.norm1 = AdaLN(embedding_size)
        self.norm2 = AdaLN(embedding_size)
        self.norm3 = AdaLN(embedding_size)

    def forward(self, src: torch.Tensor, single_eval_position: int, prob_adj, graph_embed) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        # adjacency based attention
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        # flip adjacency matrix, except for diagonal entries
        f = prob_adj.shape[0]
        eye = torch.eye(f, dtype=torch.bool)
        mask_1 = (prob_adj + eye).to(get_default_device())
        mask_2 = (prob_adj.T + eye).to(get_default_device())
        mask_3 = (torch.full((f, f), 1.0)).to(get_default_device())
        mask_1 = mask_1.unsqueeze(0).expand(self.n_feature_attn_heads[0], -1, -1)
        mask_2 = mask_2.unsqueeze(0).expand(self.n_feature_attn_heads[1], -1, -1)
        mask_3 = mask_3.unsqueeze(0).expand(self.n_feature_attn_heads[2], -1, -1)
        # mask of shape (num_heads, C, C)
        mask = torch.cat([mask_1, mask_2, mask_3], dim=0) 
        
        src = self.self_attn_graph(src, src, src, attn_mask=mask)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src, graph_embed)
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
        src = self.norm2(src, graph_embed)
        # MLP after attention
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src, graph_embed)
        return src
