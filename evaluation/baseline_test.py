from datetime import datetime
import os

from tfmplayground.utils import get_default_device
from configs.default_configs import prior_config, training_config
from graphpfn.callbacks import ValidationCallback
from graphpfn.interface import init_model_from_state_dict_file


model_paths = {'beta': 'workdir/baseline_beta_04_07_18_10/latest_checkpoint.pth',
               'uniform': 'workdir/baseline_uniform_04_07_18_17/latest_checkpoint.pth',
               'binary': 'workdir/baseline_binary_04_07_19_00/latest_checkpoint.pth',}
models = {name: init_model_from_state_dict_file(path).to(get_default_device()) for name, path in model_paths.items()}

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"evaluation/output/{datetime_str}"
os.makedirs(output_dir, exist_ok=True)

for i, mode in enumerate(["beta", "uniform", "binary"]):
    for seed in range(10):
        prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                "distribution_parameters": {"choices": [mode], "probabilities": [1.0]}}
        for name, model in models.items():
            print(f"Evaluating {name} model with seed {seed} using {mode} distribution...")
            callback = ValidationCallback(f"{output_dir}/{name}", prior_config, num_steps=100, seed=seed)
            callback.on_epoch_end(epoch= i * 10 + seed, epoch_time=0.0, loss=0.0, model=model, buckets=training_config['buckets'])
