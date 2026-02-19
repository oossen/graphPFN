import torch
from torch import nn
from graphpfn.base_model import GraphPFNModel
from graphpfn.binary_gcn_model import GCN
from tfmplayground.model import TransformerEncoderStack


class SoftGCNModel(GraphPFNModel):
    
    def _make_transformer_encoder(self) -> nn.Module:
        
        class TransformerWrapper(nn.Module):
            def __init__(self, transformer: nn.Module, gcn: nn.Module):
                super().__init__()
                self.transformer = transformer
                self.gcn = gcn

            def forward(self, x: torch.Tensor, single_eval_pos: int, **kwargs) -> torch.Tensor:
                prob_adj = kwargs['prob_adj']
                graph_embeddings = self.gcn(prob_adj)
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