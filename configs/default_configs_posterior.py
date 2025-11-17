"""
These are like the default configs, but specify a model incorporating graph knowledge for training.
"""

import configs.default_configs as defaults
from graphpfn.posterior_model import GraphPFNModel

prior_config = defaults.prior_config


training_config = defaults.training_config
training_config["model"] = model = GraphPFNModel(
    num_attention_heads=8,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=1000,
)
training_config["saveweights"] = "posterior_pfn"
