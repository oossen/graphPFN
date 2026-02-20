import configs.default_configs as defaults
from copy import deepcopy

prior_config = deepcopy(defaults.prior_config)
training_config = deepcopy(defaults.training_config)

prior_config['activations'] = prior_config['activations'][0:1]
prior_config['dataset_config']['number_train_samples_per_dataset'] = {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 50}}
prior_config['graph_config']['num_nodes'] = {"distribution": "discrete_uniform", "distribution_parameters": {"low": 3, "high": 5}}

training_config["saveweights"] = "simple"