import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

prior_config['activations'] = prior_config['activations'][:1]
prior_config['graph_config']['num_nodes'] = {'value': 3}
prior_config['noise_config']['root_std_dist'] = {'value': 1.1}
prior_config['noise_config']['non_root_std_dist'] = {'value': 0.2}