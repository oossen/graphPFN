"""
These are like the default configs, but all relevant parameters are chosen much smaller,
so that things like the training loop can be debugged easily even locally.
"""

import configs.default_configs as defaults

dataset_config = defaults.prior_config["dataset_config"]
dataset_config["number_train_samples_per_dataset"] = {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 50, "high": 100}
    }
dataset_config["number_test_samples_per_dataset"] = {"value": 10}


graph_config = defaults.prior_config["graph_config"]
graph_config["num_nodes"] = { 
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 3, "high": 5}
    }

scm_config = defaults.prior_config["scm_config"]

prior_config = {"dataset_config": dataset_config, "graph_config": graph_config, "scm_config": scm_config}


training_config = defaults.training_config
training_config["steps"] = 100
training_config["epochs"] = 10
training_config["n_bardist_samples"] = 100