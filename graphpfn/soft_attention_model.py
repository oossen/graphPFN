import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, LayerNorm

from graphpfn.base_model import GraphPFNModel
from tfmplayground.utils import get_default_device


class SoftAttentionModel(GraphPFNModel):
    def __init__(self,
                 embedding_size: int,
                 num_attention_heads: int,
                 num_feature_attention_heads: int,
                 num_graph_attention_heads: int,
                 mlp_hidden_size: int,
                 num_layers: int,
                 num_outputs: int):
        self.num_feature_attention_heads = num_feature_attention_heads
        self.num_graph_attention_heads = num_graph_attention_heads
        super().__init__(embedding_size,
                         num_attention_heads,
                         mlp_hidden_size,
                         num_layers,
                         num_outputs)
        self.architecture['num_feature_attention_heads'] = num_feature_attention_heads
        self.architecture['num_graph_attention_heads'] = num_graph_attention_heads

    def _make_transformer_encoder(self) -> nn.Module:
        encoder = TransformerEncoderStack(
            self.num_layers,
            self.embedding_size,
            self.num_attention_heads,
            self.num_feature_attention_heads,
            self.num_graph_attention_heads,
            self.mlp_hidden_size,
        )
        return encoder
    

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

    def forward(self, x: torch.Tensor, single_eval_position: int, **kwargs) -> torch.Tensor:
        prob_adj = kwargs['prob_adj']
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position, prob_adj=prob_adj)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, nhead_feature: int, nhead_graph: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True):
        super().__init__()
        self.self_attn_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first)
        # parental heads, child heads, unrestricted heads
        self.n_feature_attn_heads = (nhead_graph // 2, nhead_graph // 2, nhead_feature)
        self.self_attn_graph = MultiplicativeMultiheadAttention(embedding_size, sum(self.n_feature_attn_heads))

        self.linear1 = Linear(embedding_size, mlp_hidden_size)
        self.linear2 = Linear(mlp_hidden_size, embedding_size)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps)

    def forward(self, src: torch.Tensor, single_eval_position: int, prob_adj: torch.Tensor) -> torch.Tensor:
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
    
class MultiplicativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Linear Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask):
        """
        Args:
            query: (b, f, d)
            key: (b, f, d)
            value: (b, f, d)
            attn_mask: (f, f)

        Returns:
            attn_output: (b, f, d)
            attn_weights: (b, #heads, f, f)
        """
        b, f, d = query.size()

        # 1. Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape for multi-head attention
        q = q.view(b, f, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, f, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, f, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Calculate Scaled Dot-Product Attention Scores
        scaling = float(self.head_dim) ** -0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scaling
        
        # 4. mask in log space
        log_mask = torch.log(attn_mask + 1e-30) 
        attn_scores = attn_scores + log_mask

        # 5. Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 6. Value computation
        attn_output = attn_weights @ v

        # 7. Reshape back and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, f, self.embed_dim)
        
        output = self.out_proj(attn_output)

        return output, attn_weights