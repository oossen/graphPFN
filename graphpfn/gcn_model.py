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
        self.gcn = GCN(embedding_size, hidden_dim=embedding_size)
        self.dropout = nn.Dropout(p=0.5)

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
        graph_embeddings = self.gcn(prob_adj.float())  # (C, E)
        graph_embeddings = self.dropout(graph_embeddings)
        input += graph_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, C, E)
        # repeatedly applies the transformer block on (B,R,C,E)
        output = self.transformer_encoder(input, single_eval_pos)
        # selects the target embeddings (B,num_targets,1,E)
        output = output[:, single_eval_pos:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output


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