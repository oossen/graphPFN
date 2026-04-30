import os
from datetime import datetime

from configs.default_configs import prior_config
from graphpfn.interface import init_model_from_state_dict_file
from priors.observational_dataloader import ObservationalDataLoader
from icml_plots.eval_synthetic import metric_r2, metric_nrmse, nll_metric, evaluate_on_prior


model_names = ["baseline_beta", "attention_beta", "attention_binary", "attention_uniform"]
model_paths = {name: f"workdir/{name}/latest_checkpoint.pth" for name in model_names}
models = {name: init_model_from_state_dict_file(path) for name, path in model_paths.items()}
metrics = {"r2": metric_r2, "nrmse": metric_nrmse, "nll": nll_metric}
now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"icml_plots/output/input_mode"
os.makedirs(output_dir, exist_ok=True)
prior_config['dataset_config']['number_train_samples_per_dataset'] = {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 1, "high": 20}
        }
for uncertainty_level in ["beta", "binary", "uniform"]:
    prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                            "distribution_parameters": {"choices": [uncertainty_level], "probabilities": [1.0]}}
    prior = ObservationalDataLoader(10000, 1, prior_config, seed=42)
    evaluate_on_prior(models, prior, metrics, f"{output_dir}/results_{uncertainty_level}.csv")
