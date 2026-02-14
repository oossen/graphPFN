import configs.simple_configs as defaults
from graphpfn.binary_attention_model import BinaryAttentionModel
from copy import deepcopy

prior_config = deepcopy(defaults.prior_config)
training_config = deepcopy(defaults.training_config)


training_config["saveweights"] = "simple_binary_attention_fallback"
num_outputs = training_config["model"].num_outputs
training_config["model"] = BinaryAttentionModel(
    embedding_size=192,
    num_attention_heads=8,
    num_feature_attention_heads=4,
    num_graph_attention_heads=4,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_outputs,
)