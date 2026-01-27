import configs.likelihood_training_configs as defaults
from graphpfn.gcn_model import GraphPFNModel

prior_config = defaults.prior_config


training_config = defaults.training_config
training_config["model"] = model = GraphPFNModel(
    num_attention_heads=8,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=defaults.n_outputs,
)
training_config["saveweights"] = "likelihood_training_gcn"
