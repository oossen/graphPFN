from graphpfn.attention_model import AttentionModel
import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

training_config["saveweights"] = "attention"
num_outputs = training_config["model"].num_outputs
training_config["model"] = AttentionModel(
    embedding_size=192,
    num_attention_heads=8,
    num_feature_attention_heads=0,
    num_graph_attention_heads=8,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_outputs,
)