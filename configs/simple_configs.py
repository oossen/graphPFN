import configs.default_configs as defaults
from configs.default_configs import AsinhWrapper
from copy import deepcopy
import torch.nn as nn

prior_config = deepcopy(defaults.prior_config)
training_config = deepcopy(defaults.training_config)
    

activations = [AsinhWrapper(nn.Identity())]

prior_config['dataset_config']['number_train_samples_per_dataset'] = {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 50}}
prior_config['graph_config']['num_nodes'] = {"distribution": "discrete_uniform", "distribution_parameters": {"low": 3, "high": 5}}
prior_config['scm_config']['activations'] = {"distribution": "categorical", "distribution_parameters": {"choices": activations}}

training_config["saveweights"] = "simple"
training_config["epochs"] = 50