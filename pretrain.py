from functools import partial
from typing import List
from datetime import datetime

import torch

from nanotabpfn.evaluation import TOY_TASKS_REGRESSION
from graphpfn.callbacks import TensorboardLoggerR2Callback, EvaluationLoggerCallback, SanityCheckLoggerCallback
from nanotabpfn.model import NanoTabPFNModel
from graphpfn.train import train
from nanotabpfn.utils import get_default_device
from nanotabpfn.callbacks import Callback

from graphpfn.utils import make_bar_distribution
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from configs.default_configs import prior_config, preprocessing_config, training_config as args


device = get_default_device()
ckpt = None
if args["loadcheckpoint"]:
    ckpt = torch.load(args["loadcheckpoint"])

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                prior_config=prior_config,
                                preprocessing_config=preprocessing_config,
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
                        preprocessing_config=preprocessing_config,
                        seed=42)
dist = make_bar_distribution(prior_factory, n_buckets=args["n_buckets"], n_samples=args["n_bardist_samples"])

if ckpt:
    model.load_state_dict(ckpt['model']) 

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
tensorboard_dir =f"{args['tensorboard']}/{datetime_str}"
evaluation_callback = EvaluationLoggerCallback(tensorboard_dir, TOY_TASKS_REGRESSION, prior, dist)
sanity_callback = SanityCheckLoggerCallback(tensorboard_dir, prior, dist)
logger_callback = TensorboardLoggerR2Callback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, evaluation_callback, sanity_callback]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args["epochs"],
    accumulate_gradients=args["accumulate"],
    lr=args["lr"],
    device=device,
    callbacks=callbacks,
    ckpt=ckpt
)

torch.save(trained_model.to('cpu').state_dict(), args["saveweights"])
