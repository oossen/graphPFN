from graphpfn.gcn_model import GCNModel
import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

training_config["saveweights"] = "gcn"
num_outputs = training_config["model"].num_outputs
training_config["model"] = GCNModel(
    embedding_size=192,
    num_attention_heads=8,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_outputs,
)