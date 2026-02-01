import torch
from torch import nn
from graphpfn.base_model import GraphPFNModel
from tfmplayground.model import TransformerEncoderStack


class BaselineModel(GraphPFNModel):
    
    def _make_transformer_encoder(self) -> nn.Module:
        
        class TransformerWrapper(nn.Module):
            def __init__(self, transformer: nn.Module):
                super().__init__()
                self.transformer = transformer

            def forward(self, x: torch.Tensor, single_eval_pos: int, **kwargs) -> torch.Tensor:
                # Ignore graph information in kwargs
                return self.transformer(x, single_eval_pos)
        
        encoder = TransformerEncoderStack(
            self.num_layers,
            self.embedding_size,
            self.num_attention_heads,
            self.mlp_hidden_size,
        )
        
        return TransformerWrapper(encoder)
    
