"""
These are like the default graph configs, but all relevant parameters are chosen much smaller,
so that things like the training loop can be debugged easily even locally.
"""

import configs.debugging_configs as defaults
from graphpfn.attention_model import GraphPFNModel

prior_config = defaults.prior_config


training_config = defaults.training_config
training_config["model"] = model = GraphPFNModel(
    num_attention_heads=4,
    num_graph_attention_heads=2,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=5000,
)
training_config["saveweights"] = "attention_pfn"
