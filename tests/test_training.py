from functools import partial
from typing import List
from graphpfn.utils import make_bar_distribution

from nanotabpfn.callbacks import Callback, ConsoleLoggerCallback
from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import train

from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.debugging_configs import prior_config, preprocessing_config

num_buckets = 100
model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_buckets,
)

prior = ObservationalDataLoader(5, 10, prior_config, preprocessing_config, seed=42)

prior_factory = partial(ObservationalDataLoader,
                        batch_size=10,
                        prior_config=prior_config,
                        preprocessing_config=preprocessing_config,
                        seed=42)
dist = make_bar_distribution(prior_factory, n_buckets=num_buckets, n_samples=1000)

callbacks: List[Callback] = [ConsoleLoggerCallback()]

trained_model, loss = train(model=model, prior=prior, criterion=dist, epochs=10, callbacks=callbacks)