import os
import argparse
from typing import Dict
from itertools import islice
import torch
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from matplotlib import pyplot as plt
from pfns.bar_distribution import FullSupportBarDistribution
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.observational_dataloader_graph_prior import ObservationalDataLoader
from configs.default_configs import prior_config
from tfmplayground.utils import get_default_device
from graphpfn.interface import cross_validate

# small graphs
prior_config['graph_config']['num_nodes'] = {'value': 5}

prior = ObservationalDataLoader(num_steps=10, batch_size=1, prior_config=prior_config, seed=54)

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
dir = f"evaluation/output/{datetime_str}"
os.makedirs(dir, exist_ok=True)

fixed_positions = {
    'x0': (2, 1),
    'x1': (4, 1),
    'x2': (5, 3),
    'x3': (3, 4),
    'y': (1, 3)
}
drawing_style = {
    'node_size': 2000,
    'font_size': 20,
    'arrowsize': 35,
    'width': 3,
    'with_labels': True,
    'pos': fixed_positions
}

data = next(islice(prior, 7, None))
g = data['graph_information']['graph']
X = data['x'][0].cpu().numpy()
y = data['y'][0].cpu().numpy()
single_eval_pos = data['single_eval_pos']

nx.draw(g, **drawing_style)
plt.savefig(f"{dir}/graph.png", dpi=100)
plt.close()

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
    
for name, model in models.items():
    score = cross_validate(model, X, y, single_eval_pos, 5, **data['graph_information'])
    print(f"Model {name} scores {score}")
    
