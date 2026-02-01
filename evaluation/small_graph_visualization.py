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
prior_config['dataset_config']['number_train_samples_per_dataset'] = {'value': 10}

prior = ObservationalDataLoader(num_steps=30, batch_size=1, prior_config=prior_config, seed=66)

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
dir = f"evaluation/output/small_graph"
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
    'width': 0,
    'arrowstyle': 'simple',
    'with_labels': True,
    'pos': fixed_positions
}

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

data = next(islice(prior, 28, None))
g = data['graph_information']['graph']
prob_adj = data['graph_information']['prob_adj']
X = data['x'][0].cpu().numpy()
y = data['y'][0].cpu().numpy()
single_eval_pos = data['single_eval_pos']

nx.draw(g, **drawing_style)
plt.savefig(f"{dir}/graph.png", dpi=100)
plt.close()
for name, model in models.items():
    score = cross_validate(model, X, y, single_eval_pos, 5, **data['graph_information'])
    print(f"Model {name} scores {score}")
    
print(prob_adj)

adjacency_matrix = torch.tensor([[0, 0, 0, 0, 1],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0]])
matrices = {
    "ground_truth": prob_adj,
    "ground_truth_binary": adjacency_matrix,
    "ground_truth_binary_flipped": adjacency_matrix.T,
    "only_y_neighborhood": torch.tensor([[0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0]]),
    "weak_x0_x2": torch.tensor([[0, 0, 0, 0, 0.5],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 0.5],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0]]),
    "weak_bidirectional_x0_x2": torch.tensor([[0, 0, 0, 0, 0.5],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 0.5],
                                 [0, 0, 0, 0, 0],
                                 [0.5, 0, 0.5, 1, 0]]),
    "weak_x1": torch.tensor([[0, 0, 0, 0, 1],
                                 [0.5, 0, 0.5, 0.5, 0.5],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0]]),
    "no_x1": torch.tensor([[0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0]]),
    "no_x0": torch.tensor([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0]]),
    "x2": torch.tensor([[0, 0, 0, 0, 1],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0]]),
    "x0_flipped": torch.tensor([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0],
                                 [1, 0, 0, 1, 0]])
}

node_labels = {0: 'x0', 1: 'x1', 2: 'x2', 3: 'x3', 4: 'y'}

model = models['graph_prior']
for name, matrix in matrices.items():
    score = cross_validate(model, X, y, single_eval_pos, 5, prob_adj=matrix)
    improvement = 100 * (score / 0.8550241947174072 - 1)
    print(f"Score with matrix {name} is {score}")
    g = nx.from_numpy_array(matrix.detach().cpu().numpy(), create_using=nx.DiGraph)
    g = nx.relabel_nodes(g, node_labels)
    
    edges = g.edges()
    weights = [g[u][v]['weight'] for u, v in edges]
    base_color = (0, 0, 0) 
    edge_color = [base_color + (w,) for w in weights] 
    nx.draw(g, edge_color=edge_color, **drawing_style)
    color = 'green' if improvement > 0 else 'red'
    plt.text(x=2.2, y=2.2, s=f"Improvement:\n    {improvement:.2f}%", color=color, fontsize=24)
    plt.savefig(f"{dir}/{name}.png", dpi=100)
    plt.close()
        

    
