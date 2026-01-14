import configs.ppd_configs as defaults
from graphpfn.pos_encoding_model import GraphPFNModel


prior_config = {
    "n_train_samples": 5,
    "n_test_samples": 1,
    "fixed_graph_ratio": 0.75,
}


training_config = defaults.training_config


training_config["model"] = model = GraphPFNModel(
    num_attention_heads=8,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=200,
    )
training_config["saveweights"] = "ppd_pe_mixed"