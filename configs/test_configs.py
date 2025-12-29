import configs.default_configs as defaults

prior_config = defaults.prior_config

prior_config['graph_config']['num_nodes'] = {'value': 5}
scm_config = {'root_std': {'value': 1.0}, 'non_root_std': {'value': 1.0}}
prior_config['scm_config'] = scm_config
prior_config['dataset_config']['number_train_samples_per_dataset'] = {'value': 5}
prior_config['dataset_config']['number_test_samples_per_dataset'] = {'value': 1}

training_config = defaults.training_config

training_config["saveweights"] = "test"
training_config["batchsize"] = 1
