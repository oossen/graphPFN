import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

prior_config['activations'] = prior_config['activations'][1:2] # only ReLU
prior_config['dataset_config']['number_train_samples_per_dataset'] = {"distribution": "discrete_uniform", "distribution_parameters": {"low": 5, "high": 20}}
prior_config['dataset_config']['number_test_samples_per_dataset'] = {'value': 1}
prior_config['graph_config']['num_nodes'] = {"distribution": "discrete_uniform", "distribution_parameters": {"low": 3, "high": 6}}

training_config["saveweights"] = "simple"
training_config["epochs"] = 20