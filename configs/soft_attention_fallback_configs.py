from graphpfn.soft_attention_model import SoftAttentionModel
import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

training_config["saveweights"] = "soft_attention_fallback"
num_outputs = training_config["model"].num_outputs
training_config["model"] = SoftAttentionModel(
    embedding_size=192,
    num_attention_heads=8,
    num_feature_attention_heads=4,
    num_graph_attention_heads=4,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_outputs,
)