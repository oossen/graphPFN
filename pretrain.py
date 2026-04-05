import argparse
from typing import List
from datetime import datetime
from pathlib import Path
import torch
from copy import deepcopy

from graphpfn.callbacks import ValidationCallback
from graphpfn.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import Callback, TensorboardLoggerCallback

from visualization.make_visualization import plot_all

from priors.observational_dataloader import ObservationalDataLoader
from configs.attention_configs import prior_config, training_config as args


prior_config = deepcopy(prior_config)
args = deepcopy(args)

# argparse setup
parser = argparse.ArgumentParser()
parser.add_argument("--prob_adj_mode", choices=["binary", "beta", "uncertain"], required=True,
                    help="Select probabilistic adjacency matrix creation mode from [binary, beta, uncertain].")
cmd_args = parser.parse_args()
prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                "distribution_parameters": {"choices": [cmd_args.prob_adj_mode], "probabilities": [1.0]}}
args["saveweights"] += f"_{cmd_args.prob_adj_mode}"

device = get_default_device()
seed = 42

ckpt_path = f"workdir/{args['saveweights']}/latest_checkpoint.pth"
if Path(ckpt_path).exists():
    seed = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False).get('seed', seed)
    seed += 1 # to ensure different data is sampled if we are continuing from a previous checkpoint
else:
    ckpt_path = None

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                prior_config=prior_config,
                                seed=seed)

model = args["model"]

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
run_name = f"{args['saveweights']}_{datetime_str}"
output_dir = f"workdir/{run_name}"
tensorboard_dir = f"{output_dir}/tensorboard"
validation_callback = ValidationCallback(tensorboard_dir, prior_config)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, validation_callback]

visualization_prior = ObservationalDataLoader(20, 1, prior_config=prior.prior_config, seed=seed)
plot_all(visualization_prior, f"{output_dir}/visualization")
    
trained_model, loss = train(
    model=model,
    prior=prior,
    buckets=args["buckets"].to(device),
    epochs=args["epochs"],
    lr=args["lr"],
    accumulate_gradients=args["accumulate_gradients"],
    callbacks=callbacks,
    run_name=run_name,
    ckpt_path=ckpt_path,
)
