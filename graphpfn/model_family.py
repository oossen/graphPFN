import torch
from torch import nn
from tfmplayground.model import NanoTabPFNModel


class GraphPFNModel(NanoTabPFNModel):
    def __init__(self,
                 family_size: int,
                 embedding_size: int,
                 num_attention_heads: int,
                 mlp_hidden_size: int,
                 num_layers: int,
                 num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        nn.Module.__init__(self)
        self.family_size = family_size
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        
        self.models = nn.ModuleList()
        for _ in range(family_size):
            self.models.append(NanoTabPFNModel(embedding_size, num_attention_heads, mlp_hidden_size, num_layers, num_outputs))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        model_index = kwargs['graph_index']
        return self.models[model_index](*args, **kwargs)