import math
import os
from dopfnprior.scm.scm import SCM
from dopfnprior.scm.scm_builder import SCMBuilder
from dopfnprior.utils.sampling import build_samplers, sample_parameters
from dopfnprior.causal_graph.graph_builder import GraphBuilder
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Dict, Any, Mapping
import numpy as np
import torch
from dopfnprior.mechanisms.simple_mechanism import SimpleMechanism
import networkx as nx
from scipy.integrate import quad

FIXED_POS = {
    'x0': (2, 1),
    'x1': (4, 1),
    'x2': (5, 3),
    'x3': (3, 4),
    'y': (1, 3)
}
EXTRA_POS = {
    'x0': (2, 0.5),
    'x1': (4, 0.5),
    'x2': (6, 3),
    'x3': (3, 4.5),
    'y': (0, 3)
}
DRAWING_STYLE = {
    'node_size': 1000,
    'font_size': 10,
    'arrowsize': 10,
    'width': 0,
    'arrowstyle': 'simple',
    'with_labels': True,
    'pos': FIXED_POS
}


def plot_likelihood(scm: SCM, values: Dict[Any, Tensor], y_var: str, filename: str, steps=200, calculate_total_mass=False):
    os.makedirs(filename, exist_ok=True)
    shape = values[list(values.keys())[0]].shape
    for i, idx in enumerate(np.ndindex(shape)):
        values_i = {v: values[v][idx] for v in values}
        
        def conditional_log_likelihood(yi):
            # yi needs to be in the same format as the other values
            return scm.log_likelihood(values_i | {y_var: torch.tensor([[yi]], dtype=torch.float32)}, y_var)
        
        # adaptively find good range for y
        y_explore = np.linspace(-10.0, 10.0, steps)
        p_explore = np.array([math.exp(conditional_log_likelihood(yi)) for yi in y_explore])
        eps = 1e-3
        mask = p_explore > eps
        indices = np.where(mask)[0]
        buffer = 1
        start_idx = max(0, indices[0] - buffer)
        end_idx = min(len(y_explore) - 1, indices[-1] + buffer)
        a = y_explore[start_idx]
        b = y_explore[end_idx]
        y = np.linspace(a, b, steps)

        if calculate_total_mass:
            total_probability_mass = quad(lambda yi: math.exp(conditional_log_likelihood(yi)), a, b)[0]
            print(f"Total probability mass on [{a}, {b}]: {total_probability_mass}")

        log_p = [conditional_log_likelihood(yi) for yi in y]
        plt.plot(y, log_p, label="log p(y)")
        plt.axhline(0, color='black', linewidth=0.5) # Adds x-axis
        plt.axvline(x=values_i['y'].item(), color='red', linestyle='--', linewidth=1)  # the true y-value
        plt.xlabel("y")
        plt.ylabel("log p(y)")
        plt.title(f"Plot of log p(y) on [{a}, {b}]")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filename}/log_likelihood_{i}.png", dpi=300)
        plt.close()
        
        p = [math.exp(lp) for lp in log_p]
        plt.plot(y, p, label="p(y)")
        plt.axvline(x=values_i['y'].item(), color='red', linestyle='--', linewidth=1)  # the true y-value
        plt.xlabel("y")
        plt.ylabel("p(y)")
        plt.title(f"Plot of p(y) on [{a}, {b}]")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filename}/likelihood_{i}.png", dpi=300)
        plt.close()
        
        # extract weights and activations
        weights = {}
        extra_labels = {}
        nodes = scm.dag.nodes
        mechs: Mapping = scm.mechanisms
        for v in nodes:
            activation = str(mechs[v].activation._module[1])
            noise_std = f"{scm.noise[v].distribution.scale:.2f}"
            extra_labels[v] = f"{activation}\nNoise σ={noise_std}"
            for w in scm.dag.predecessors(v):
                weight = mechs[v].weights[w].item()
                weights[(w, v)] = f"{weight:.2f}"
        nx.draw(scm.dag, **DRAWING_STYLE)
        nx.draw_networkx_edge_labels(scm.dag, FIXED_POS, edge_labels=weights)
        nx.draw_networkx_labels(scm.dag, EXTRA_POS, labels=extra_labels)
        plt.margins(0.5)
        plt.savefig(f"{filename}/graph_{i}.png", dpi=300)
        plt.close()
        
        # save values
        with open(f"{filename}/values_{i}.txt", "w") as f:
            for v in values_i:
                f.write(f"{v}: {values_i[v].item()}\n")
    

if __name__ == "__main__":
    from configs.ppd_configs import prior_config
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    
    generator = torch.Generator()
    generator.manual_seed(43)
        
    graph_samplers = build_samplers(prior_config['graph_config'], "graph")
    scm_samplers = build_samplers(prior_config['scm_config'], "scm")
    graph_params = sample_parameters(graph_samplers, generator)
    scm_params = sample_parameters(scm_samplers, generator)
    
    for i in range(10):
        graph_builder = GraphBuilder(**graph_params)
        graph = graph_builder.sample(generator)
        scm_builder = SCMBuilder(graph, **scm_params)
        scm = scm_builder.sample(generator)
        sample_shape = (1,)
        scm.sample_noise(sample_shape, generator=generator)
        values = scm.propagate(sample_shape)
        
        plot_likelihood(scm, values, 'y', f"visualization/output/{datetime_str}/{i}")