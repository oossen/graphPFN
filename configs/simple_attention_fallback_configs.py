import configs.simple_configs as defaults
from graphpfn.attention_model import AttentionModel
from copy import deepcopy

prior_config = deepcopy(defaults.prior_config)
training_config = deepcopy(defaults.training_config)


training_config["saveweights"] = "simple_attention_fallback"
num_outputs = training_config["model"].num_outputs
training_config["model"] = AttentionModel(
    embedding_size=192,
    num_attention_heads=8,
    num_feature_attention_heads=4,
    num_graph_attention_heads=4,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_outputs,
)