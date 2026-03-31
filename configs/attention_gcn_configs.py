from graphpfn.attention_gcn_model import AttentionGCNModel
import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

training_config["saveweights"] = "attention_gcn"
num_outputs = training_config["model"].num_outputs
training_config["model"] = AttentionGCNModel(
    embedding_size=192,
    num_attention_heads=8,
    num_feature_attention_heads=0,
    num_graph_attention_heads=8,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_outputs,
)