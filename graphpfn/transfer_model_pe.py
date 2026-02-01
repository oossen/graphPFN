from typing import Tuple
import torch
import torch.nn as nn
from tfmplayground.model import NanoTabPFNModel
from graphpfn.pos_encoding_model import Decoder, TransformerEncoderLayer, get_positional_encoding

class GraphPFNModel(nn.Module):
    def __init__(self, embedding_size: int, 
                 num_attention_heads: int, 
                 mlp_hidden_size: int, 
                 num_layers: int, num_outputs: int, 
                 pretrained_model: str = "ppd_01_11_16_10", 
                 model_class = NanoTabPFNModel):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        
        # load frozen backbone
        state_dict = torch.load(f"workdir/{pretrained_model}/latest_checkpoint.pth", map_location=torch.device('cpu'))
        model = model_class(**state_dict['architecture'])
        model.load_state_dict(state_dict['model'])
        self.backbone = model
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # decoder
        self.hot_trafo_layer = TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        
    def forward(self, *args, **kwargs):
        if len(args) == 3:
            # case model(train_x, train_y, test_x)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), single_eval_pos=None)
            return self._forward(*args, **kwargs)

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, **kwargs) -> torch.Tensor:
        x_src, y_src = src
    
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
            
        # converts scalar values to embeddings (B, R, C-1, E)
        x_src = self.backbone.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        
        # converts scalar values to embeddings (B, R, 1, E)
        y_src = self.backbone.target_encoder(y_src, num_rows)
        
        # Concatenates to give (B, R, C, E)
        input = torch.cat([x_src, y_src], 2)
        
        # --- ADD POSITIONAL ENCODING HERE ---
        # B = Batches, R = Rows, C = Columns, E = Embedding Dim
        B, R, C, E = input.shape
        
        # Generate the PE for the column dimension C
        pe = get_positional_encoding(C, E, input.device)

        # repeatedly applies the transformer block on (B, R, C, E)
        output = self.backbone.transformer_encoder(input, single_eval_pos)
        
        output = output + pe
        
        # now the unfrozen part
        output = self.hot_trafo_layer(output, single_eval_pos)
        
        # selects the target embeddings (B, num_targets, 1, E)
        output = output[:, single_eval_pos:, -1, :]
        
        # runs the embeddings through the decoder
        output = self.decoder(output)
        return output