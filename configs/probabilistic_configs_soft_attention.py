from graphpfn.graph_prior_model import GraphPFNModel
import configs.probabilistic_configs as defaults


prior_config = defaults.prior_config


training_config = defaults.training_config
n_outputs = training_config["model"].num_outputs
training_config["model"] = model = GraphPFNModel(
    num_attention_heads=8,
    num_graph_attention_heads=4,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=n_outputs,
)
training_config["saveweights"] = "probabilistic_soft_attention"