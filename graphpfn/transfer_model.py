import torch
import torch.nn as nn
from tfmplayground.model import NanoTabPFNModel

class TransferLearningModel(nn.Module):
    def __init__(self, pretrained_model: str, model_class = NanoTabPFNModel):
        super().__init__()
        
        # load frozen backbone
        state_dict = torch.load(f"workdir/{pretrained_model}/latest_checkpoint.pth", map_location=torch.device('cpu'))
        model = model_class(**state_dict['architecture'])
        model.load_state_dict(state_dict['model'])
        self.backbone = model
        for param in self.backbone.parameters():
            param.requires_grad = False