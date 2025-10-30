from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from nanotabpfn.model import Decoder, FeatureEncoder, TargetEncoder, NanoTabPFNModel, TransformerEncoderStack
from nanotabpfn.utils import get_default_device


class GraphPFNModel(NanoTabPFNModel):
    def __init__(self,
                 embedding_size: int,
                 num_attention_heads: int,
                 gcn_hidden_size: int,
                 mlp_hidden_size: int,
                 num_layers: int,
                 num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        nn.Module.__init__(self)
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.gcn_hidden_size = gcn_hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_encoder = TransformerEncoderStack(num_layers, embedding_size, num_attention_heads, mlp_hidden_size)
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        self.graph_encoder = GraphEncoder(3, gcn_hidden_size, embedding_size)

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

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, adjacency_matrix: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        x_src, y_src = src
        if adjacency_matrix is None:
            num_cols = x_src.shape[2] + 1
            adjacency_matrix = torch.full((num_cols, num_cols), 0.5)

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
        # add adjacency matrix encoding
        # encoding has shape (C, E) before unsqueezing, (1, 1, C, E) after
        adj_encoding = self.graph_encoder(adjacency_matrix).unsqueeze(0).unsqueeze(0)
        input += adj_encoding
        # repeatedly applies the transformer block on (B,R,C,E)
        output = self.transformer_encoder(input, single_eval_pos)
        # selects the target embeddings (B,num_targets,1,E)
        output = output[:, single_eval_pos:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output
    

class GraphEncoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gcn_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNLayer(hidden_dim, out_dim))
        
    def forward(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # initial encoding, like a positional encoding
        f = adjacency_matrix.shape[0]
        pe = torch.zeros(f, self.hidden_dim)
        pos = torch.arange(0, f).unsqueeze(1)
        i = torch.arange(0, self.hidden_dim).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / self.hidden_dim)
        angles = pos * angle_rates
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        
        x = pe
        for layer in self.gcn_layers:
            x = layer(x, adjacency_matrix)
        return x

    
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features, device=get_default_device()))
        self.bias = nn.Parameter(torch.empty(out_features, device=get_default_device()))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adjacency_matrix):
        input = input.to(get_default_device())
        adj = adjacency_matrix.to(get_default_device())
        I = torch.eye(adj.size(0), device=get_default_device())
        adj_hat = adj + I
        D_hat = torch.diag(torch.pow(adj_hat.sum(1), -0.5))
        adj_norm = D_hat @ adj_hat @ D_hat

        output = adj_norm @ input @ self.weight + self.bias
        return F.relu(output)
    
