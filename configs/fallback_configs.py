import configs.default_configs as defaults
from graphpfn.fallback_model import GraphPFNModel

prior_config = defaults.prior_config


training_config = defaults.training_config
training_config["model"] = model = GraphPFNModel(
    num_attention_heads=8,
    num_feature_attention_heads=12,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=1000,
)
training_config["saveweights"] = "fallback"
