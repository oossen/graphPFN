import configs.ppd_configs as defaults
from graphpfn.binary_mask_model import GraphPFNModel

prior_config = defaults.prior_config
training_config = defaults.training_config


training_config["model"] = model = GraphPFNModel(
    num_attention_heads=8,
    num_graph_attention_heads=4,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=300,
    )
training_config["saveweights"] = "ppd_graph"