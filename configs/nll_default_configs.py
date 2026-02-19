import configs.default_configs as defaults

prior_config = defaults.prior_config
training_config = defaults.training_config

training_config["saveweights"] = "nll_baseline"
training_config["nll"] = True