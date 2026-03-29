from typing import Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from tfmplayground.model import Decoder, TargetEncoder, FeatureEncoder


@dataclass(eq=False)
class GraphPFNModel(nn.Module, ABC):
    embedding_size: int
    num_attention_heads: int
    mlp_hidden_size: int
    num_layers: int
    num_outputs: int
    
    def __post_init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder(self.embedding_size)
        self.target_encoder = TargetEncoder(self.embedding_size)
        self.transformer_encoder = self._make_transformer_encoder()
        self.decoder = Decoder(self.embedding_size, self.mlp_hidden_size, self.num_outputs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Like NanoTabPFN, provides two interfaces:
        
        model(X_train, y_train, X_test, **kwargs)
            Args:
                X_train : (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, num_features)
                y_train : (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                X_test : (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_features)
                **kwargs : additional arguments providing graph information

        model((X, y), single_eval_pos, **kwargs)
            Args:
                X: (torch.Tensor) a tensor of shape (batch_size, num_datapoints, num_features)
                y: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                **kwargs : additional arguments providing graph information
                single_eval_pos: int

        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_classes), which represent the predicted logits.
        """
        if len(args) == 3:
            # case model(X_train, y_train, X_test, **kwargs)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((X, y), single_eval_pos, **kwargs)
            return self._forward(*args, **kwargs)
        else:
            raise ValueError("Invalid input!")

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, **kwargs) -> torch.Tensor:
        x_src, y_src = src
        # (B, single_eval_pos) -> (B, single_eval_pos, 1)
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # (B, R, C-1) -> (B, R, C-1, E)
        x_src = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        # (B, single_eval_pos, 1) -> (B, R, 1, E)
        y_src = self.target_encoder(y_src, num_rows)
        # (B, R, C-1, E) + (B, R, 1, E) -> (B, R, C, E)
        input = torch.cat([x_src, y_src], 2)
        # transformer backbone
        output = self.transformer_encoder(input, single_eval_pos, **kwargs)
        # (B, R, C, E) -> (B, R-single_eval_pos, E)
        output = output[:, single_eval_pos:, -1, :]
        # (B, R-single_eval_pos, E) -> (B, R-single_eval_pos, num_outputs)
        output = self.decoder(output)
        return output
    
    @abstractmethod
    def _make_transformer_encoder(self) -> nn.Module:
        pass