from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tfmplayground.utils import get_default_device

from tfmplayground.model import Decoder, FeatureEncoder, TargetEncoder, NanoTabPFNModel, TransformerEncoderStack


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
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

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
            if 'prob_adj' in kwargs:
                return self._forward(*args, **kwargs)
            elif 'adjacency_matrix' in kwargs:
                kwargs['prob_adj'] = kwargs['adjacency_matrix']
                return self._forward(*args, **kwargs)
            else:
                x_src, y_src = args[0]
                num_cols = x_src.shape[2] + 1
                kwargs['prob_adj'] = torch.full((num_cols, num_cols), 0.5)
                return self._forward(*args, **kwargs)
        else:
            raise ValueError("Invalid input!")

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, prob_adj: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        y_src = self.target_encoder(y_src, num_rows)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        input = torch.cat([x_src, y_src], 2)
        # add graph encoding
        graph_embeddings = power_series_dag_embedding(prob_adj, self.embedding_size)  # (C, E)
        input += graph_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, C, E)
        # repeatedly applies the transformer block on (B,R,C,E)
        output = self.transformer_encoder(input, single_eval_pos)
        # selects the target embeddings (B,num_targets,1,E)
        output = output[:, single_eval_pos:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output


def power_series_dag_embedding(adj, dim, alpha=0.5, max_depth=3, seed=42):
    """
    Embeds a DAG using a power series of its adjacency matrix with a fixed seed.
    """
    adj = adj.to(get_default_device())
    
    N = adj.size(0)
    device = adj.device
    if max_depth is None:
        max_depth = N 

    # Use a local generator to ensure R is deterministic based on the seed
    # This avoids messing with the global torch.manual_seed()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    # Initialize Random Projection Matrix R ~ N(0, 1/D)
    # R is the "base" mapping; keeping this fixed ensures nodes 
    # are projected into a consistent latent space.
    R = torch.randn(N, dim, device=device, generator=gen) / (dim ** 0.5)
    
    embedding = R.clone()
    current_term = R
    
    for k in range(1, max_depth):
        current_term = torch.mm(adj, current_term)
        
        if torch.norm(current_term) < 1e-9:
            break
            
        embedding += (alpha ** k) * current_term
        
    return embedding