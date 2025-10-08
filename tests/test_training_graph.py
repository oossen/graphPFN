from typing import List
from graphpfn.model import GraphPFNModel
from graphpfn.utils import make_bar_distribution

from nanotabpfn.callbacks import Callback, ConsoleLoggerCallback
from graphpfn.train import train

from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.debugging_configs import graph_config, scm_config, preprocessing_config, dataset_config

num_buckets = 100
model = GraphPFNModel(
    num_attention_heads=6,
    num_graph_attention_heads=4,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_buckets,
)

prior = ObservationalDataLoader(5, 10, graph_config, scm_config, preprocessing_config, dataset_config, seed=42)

bar_dist_prior = ObservationalDataLoader(1000, 1, graph_config, scm_config, preprocessing_config, dataset_config, seed=42)
dist = make_bar_distribution(bar_dist_prior, n_buckets=num_buckets, n_samples=10000)

callbacks: List[Callback] = [ConsoleLoggerCallback()]

trained_model, loss = train(model=model, prior=prior, criterion=dist, epochs=10, callbacks=callbacks)