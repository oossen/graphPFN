from typing import Tuple
import torch
from torch import nn

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
            return self._forward(*args, **kwargs)
        else:
            raise ValueError("Invalid input!")

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, adjacency_matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        x_src, y_src = src
        # compute Markov blanket
        blanket = markov_blanket(adjacency_matrix)
        x_src = x_src[:, :, blanket]
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
    
    
def markov_blanket(A):
    """Return the Markov blanket of the final node wrt `A`."""
    n = A.shape[0]
    target = n - 1

    # parents: nodes with edges -> target
    parents = (A[:, target] == 1).nonzero(as_tuple=True)[0]

    # children: nodes with edges from target ->
    children = (A[target] == 1).nonzero(as_tuple=True)[0]

    # co-parents: parents of the children (excluding target)
    coparents = []
    for c in children:
        p = (A[:, c] == 1).nonzero(as_tuple=True)[0]
        p = p[p != target]
        coparents.append(p)
    if coparents:
        coparents = torch.unique(torch.cat(coparents))
    else:
        coparents = torch.tensor([], dtype=torch.long)

    blanket = torch.unique(torch.cat([parents, children, coparents]))
    return blanket