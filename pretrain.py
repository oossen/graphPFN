from functools import partial
from typing import List
from datetime import datetime

import torch

from nanotabpfn.evaluation import TOY_TASKS_REGRESSION
from graphpfn.callbacks import TensorboardLoggerR2Callback, EvaluationLoggerCallback, SanityCheckLoggerCallback
from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import train
from nanotabpfn.utils import get_default_device
from nanotabpfn.callbacks import Callback

from graphpfn.utils import make_bar_distribution
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from configs.default_configs import prior_config, training_config as args
from visualization.make_visualization import make_all


device = get_default_device()

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                prior_config=prior_config,
                                seed=42)

model = NanoTabPFNModel(
    num_attention_heads=args["heads"],
    embedding_size=args["embeddingsize"],
    mlp_hidden_size=args["hiddensize"],
    num_layers=args["layers"],
    num_outputs=args["n_buckets"],
)

prior_factory = partial(ObservationalDataLoader,
                        batch_size=10,
                        prior_config=prior_config,
                        seed=42)
dist = make_bar_distribution(prior_factory, n_buckets=args["n_buckets"], n_samples=args["n_bardist_samples"])

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
tensorboard_dir = f"{args['output']}/{datetime_str}/tensorboard"
evaluation_callback = EvaluationLoggerCallback(tensorboard_dir, TOY_TASKS_REGRESSION, prior, dist)
sanity_callback = SanityCheckLoggerCallback(tensorboard_dir, prior, dist)
logger_callback = TensorboardLoggerR2Callback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, evaluation_callback, sanity_callback]

# visualize data and save configs
make_all(prior_config, f"{args['output']}/{datetime_str}/visualization")

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args["epochs"],
    accumulate_gradients=args["accumulate"],
    lr=args["lr"],
    device=torch.device(device),
    callbacks=callbacks,
)

torch.save(trained_model.to('cpu').state_dict(), args["saveweights"])
