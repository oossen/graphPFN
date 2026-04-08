from graphpfn.baseline_model import BaselineModel
import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

training_config["saveweights"] = "baseline_fallback"
num_outputs = training_config["model"].num_outputs
training_config["model"] = BaselineModel(
        num_attention_heads=12,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=num_outputs,
    )