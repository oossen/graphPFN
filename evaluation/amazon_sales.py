import os
import argparse
from typing import Dict
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from pfns.bar_distribution import FullSupportBarDistribution
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.observational_dataloader_graph_prior import ObservationalDataLoader
from configs.default_configs import prior_config
from tfmplayground.utils import get_default_device
from graphpfn.interface import cross_validate

df = pd.read_csv('evaluation/2021 Data.csv', index_col='Date')
df = df.astype(float)
y = df['Profit'].to_numpy()
X = df.drop('Profit', axis=1).to_numpy()
single_eval_pos = 300

# features
# 0               1        2          3          4          5       6                7
# Shopping Event? Ad Spend Page Views Unit Price Sold Units Revenue Operational Cost Profit
adjacency_matrix = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0]])

model_paths = {'pfn': 'workdir/nano_tab_pfn',
              'graph_prior': 'workdir/graph_prior',
              'binary': 'workdir/binary',
              'fallback': 'workdir/nano_tab_pfn_fallback',
              'graph_prior_fallback': 'workdir/graph_prior_fallback',
              'binary_fallback': 'workdir/binary_fallback'}
models = {}
for name, path in model_paths.items():
    model = init_model_from_state_dict_file(name, f"{path}/latest_checkpoint.pth")
    buckets = torch.load(f"{path}/dist.pth")
    dist = FullSupportBarDistribution(buckets)
    reg = Regressor(model, dist, get_default_device())
    models[name] = reg
    
for name, model in models:
        score = cross_validate(model, X, y, single_eval_pos, 5, adjacency_matrix=adjacency_matrix)
        print(f"Score for {name}: R² = {score}")
    
