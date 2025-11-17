from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, LayerNorm
import networkx as nx

from tfmplayground.model import Decoder, FeatureEncoder, TargetEncoder, NanoTabPFNModel, TransformerEncoderStack
from tfmplayground.utils import get_default_device

from dopfnprior.scm.scm import SCM
from dopfnprior.mechanisms.base_mechanism import BaseMechanism


class GraphPFNModel(NanoTabPFNModel):
    def __init__(self,
                 embedding_size: int,
                 num_attention_heads: int,
                 mlp_hidden_size: int,
                 num_layers: int,
                 num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        nn.Module.__init__(self)
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_encoder = TransformerEncoderStack(num_layers, embedding_size, num_attention_heads, mlp_hidden_size)
        self.weight_decoder = Decoder(embedding_size, mlp_hidden_size, 1)
        self.bias_decoder = Decoder(embedding_size, mlp_hidden_size, 1)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Like NanoTabPFN, provides two interfaces:
        
        model(X_train, y_train, X_test, adjacency_matrix=adjacency_matrix)
            Args:
                X_train : (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, num_features)
                y_train : (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                X_test : (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_features)
                adjacency_matrix : (torch.Tensor) a tensor of shape (num_features+1, num_features+1)

        model((x, y), single_eval_pos, adjacency_matrix=adjacency_matrix)
            Args:
                x: (torch.Tensor) a tensor of shape (batch_size, num_datapoints, num_features)
                y: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                adjacency_matrix : (torch.Tensor) a tensor of shape (num_features+1, num_features+1)
                single_eval_pos: int

        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_classes),
                           which represent the predicted logits
        """
        if len(args) == 3:
            # case model(train_x, train_y, test_x, adjacency_matrix=adjacency_matrix)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), single_eval_pos=single_eval_pos, adjacency_matrix=adjacency_matrix)
            return self._forward(*args, **kwargs)
        else:
            raise ValueError("Invalid input!")

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, new_graph: nx.DiGraph, **kwargs) -> torch.Tensor:
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src_enc = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        y_src_enc = self.target_encoder(y_src, num_rows)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        input = torch.cat([x_src_enc, y_src_enc], 2)
        # repeatedly applies the transformer block on (B,R,C,E)
        encoding = self.transformer_encoder(input, single_eval_pos)
        # predict mechanism
        parents = new_graph.predecessors('y')
        summands = []
        for v in parents:
            idx = int(v[1]) # v='xi', so int(v[1])=i
            parent_embedding = encoding[:, :, idx, :]
            weight = self.weight_decoder(parent_embedding)
            weight = torch.mean(weight, dim=1, keepdim=True).squeeze(-1) # (B, 1)
            summand = x_src[:, single_eval_pos:, idx] * weight # (B, N - single_eval_pos)
            summands.append(summand)
        bias = self.bias_decoder(encoding)
        bias = torch.mean(bias, dim=[1, 2]).squeeze(-1)
        # activation
        activation = nn.ReLU()
        output = activation(sum(summands) + bias.unsqueeze(-1)) 
        return output
    

class DeterministicMechanism(BaseMechanism):
    def __init__(self, weights: torch.Tensor, biases: torch.Tensor, *, input_dim: int, node_dim: int = 1) -> None:
        super().__init__(input_dim=input_dim, node_dim=node_dim)
        self.weights = weights
        self.biases = biases
        self.activation = nn.ReLU()
        
    def _forward(self, parents: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        # both self.weights and parents have shape (b, n, f)
        out = (self.weights * parents).sum(dim=-1, keepdim=True) + self.biases
        out = self.activation(out)
        return out