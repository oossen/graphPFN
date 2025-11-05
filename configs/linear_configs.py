"""
Training config for the linear dataloader.
"""

import configs.default_configs as defaults
from graphpfn.model_family import GraphPFNModel


training_config = defaults.training_config

training_config["model"] = GraphPFNModel(
        family_size=3,
        num_attention_heads=8,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=100,
    )