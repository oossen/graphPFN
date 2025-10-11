from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, LayerNorm

from nanotabpfn.model import Decoder, FeatureEncoder, TargetEncoder, NanoTabPFNModel
from nanotabpfn.utils import get_default_device


class GraphPFNModel(NanoTabPFNModel):
    def __init__(self,
                 embedding_size: int,
                 num_attention_heads: int,
                 num_graph_attention_heads: int,
                 mlp_hidden_size: int,
                 num_layers: int,
                 num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        nn.Module.__init__(self)
        self.num_outputs = num_outputs
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_encoder = TransformerEncoderStack(num_layers, embedding_size, num_attention_heads, num_graph_attention_heads, mlp_hidden_size)
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Like nanoTabPFN, provides two interfaces:
        
        model(X_train, y_train, X_test, adjacency_matrix)
            Args:
                X_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, num_features)
                y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                X_test: (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_features)

        model((x,y), single_eval_pos)
            Args:
                x: (torch.Tensor) a tensor of shape (batch_size, num_datapoints, num_features)
                y: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                single_eval_pos: int

        Specify an adjacency matrix using a keyword argument:
            adjacency_matrix: (torch.Tensor) a binary tensor of shape (num_features + 1, num_features + 1)

        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_classes),
                           which represent the predicted logits
        """
        if len(args) == 3:
            # case model(train_x, train_y, test_x, adjacency_matrix)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), single_eval_pos=None)
            return self._forward(*args, **kwargs)
        else:
            raise ValueError("Invalid input!")

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, adjacency_matrix: torch.Tensor | None = None) -> torch.Tensor:
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
        # repeatedly applies the transformer block on (B,R,C,E)
        output = self.transformer_encoder(input, single_eval_pos, adjacency_matrix)
        # selects the target embeddings (B,num_targets,1,E)
        output = output[:, single_eval_pos:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output
    

class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, num_attention_heads: int, num_graph_attention_heads: int, mlp_hidden_size: int):
        """ Instantiates num_layers many Transformer Blocks and stores them in a list so we can use them in the forward """
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, num_graph_attention_heads, mlp_hidden_size))

    def forward(self, x: torch.Tensor, single_eval_position: int, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Takes the embeddings of all the cells of the table as input and applies num_layers many Transformer blocks.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                              for all the cells in the table
            single_eval_position: (int) the length of X_train
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position, adjacency_matrix=adjacency_matrix)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """

    def __init__(self, embedding_size: int, nhead: int, nhead_graph: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attn_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attn_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attn_graph_parents = MultiheadAttention(embedding_size, nhead_graph, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attn_graph_children = MultiheadAttention(embedding_size, nhead_graph, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm4 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, single_eval_position: int, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                                for all the cells in the table
            single_eval_position: (int) the length of X_train
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src = self.self_attn_between_features(src, src, src)[0]+src
        src = self.norm1(src)
        # adjacency based attention
        mask = (1 - adjacency_matrix).to(get_default_device())
        src = self.self_attn_graph_parents(src, src, src, attn_mask=mask)[0]+src
        src = self.self_attn_graph_children(src, src, src, attn_mask=mask.T)[0]+src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm2(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attn_between_datapoints(src[:,:single_eval_position], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        # test data attends to the training data
        src_right = self.self_attn_between_datapoints(src[:,single_eval_position:], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        src = torch.cat([src_left, src_right], dim=1)+src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm3(src)
        # MLP after attention
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm4(src)
        return src