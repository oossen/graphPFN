import math
import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from datetime import datetime
from pfns.bar_distribution import FullSupportBarDistribution
from dopfnprior.scm.scm import SCM
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.basic_dataloader import ObservationalDataLoader
from visualization.plotting import plot_graph
from tfmplayground.utils import get_default_device


EPS = 1e-2

FIXED_POS = {
    'x0': (2, 1),
    'x1': (4, 1),
    'x2': (5, 3),
    'x3': (3, 4),
    'y': (1, 3)
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


@torch.no_grad()
def mcmc(values: Dict, prior: ObservationalDataLoader, generator: torch.Generator) -> List[Tuple[SCM, float]]:
    """Perform basic MCMC where the proposal distribution is just the prior."""
    incumbent_log_prob = float("-inf")
    chain = []
    for i, data in enumerate(prior):
        print(f"MCMC step {i+1}/{len(prior)}...")
        proposal: SCM = data['graph_information']['scm']
        proposal_log_prob = proposal.log_likelihood(values)
        accept = False
        if proposal_log_prob > incumbent_log_prob:
            print(f"Improving move... ({incumbent_log_prob} -> {proposal_log_prob})")
            accept = True
        else:
            print(f"Non-improving move... ({incumbent_log_prob} -> {proposal_log_prob})")
            acceptance_ratio = math.exp(proposal_log_prob - incumbent_log_prob)
            if torch.rand((1,), generator=generator) < acceptance_ratio:
                accept = True
        if accept:
            print("Move accepted...")
            chain.append((proposal, proposal_log_prob))
            incumbent_log_prob = proposal_log_prob
    return chain


def ppd(values: Dict, samples: List[SCM], y: torch.Tensor) -> torch.Tensor:
    """
    Approximate the PPD of y given the features in `values`
    using Monte Carlo integration with the provided `samples`.
    """
    likelihoods = []
    for scm in samples:
        ll = scm.log_likelihood_batch(values, y)
        likelihoods.append(torch.exp(ll))
    return torch.stack(likelihoods).mean(dim=0)


def plot_ppd(ax, values: Dict, samples: List[SCM], style: Dict, steps: int = 100):
    # Find good range for y
    y_explore = torch.linspace(-10.0, 10.0, steps)
    p_explore = torch.stack([torch.exp(scm.log_likelihood_batch(values, y_explore)) for scm in samples]).mean(dim=0)
    mask = p_explore > EPS
    indices = torch.where(mask)[0]
    start_idx = max(0, indices[0])
    end_idx = min(len(y_explore) - 1, indices[-1])
    a = y_explore[start_idx]
    b = y_explore[end_idx]
    # Plot and change range
    y = torch.linspace(a, b, steps)
    likelihoods = [torch.exp(scm.log_likelihood_batch(values, y)) for scm in samples]
    for ll in likelihoods:
        ax.plot(y, ll, **{k: v for k, v in style.items() if k != 'label'}, alpha=0.02)
    probs = torch.stack(likelihoods).mean(dim=0)
    ax.plot(y, probs, **style)
    curr_min, curr_max = ax.get_xlim()
    ax.set_xlim(min(curr_min, a), max(curr_max, b))
    

def plot_ppd_pfn(ax, values: Dict, X_train, y_train, model_path: str, model_type: str, style: Dict, steps: int = 100, **kwargs):
    # Initialize and fit model
    model = init_model_from_state_dict_file(model_type, f"{model_path}/latest_checkpoint.pth")
    buckets = torch.load(f"{model_path}/dist.pth")
    bar_dist = FullSupportBarDistribution(buckets)
    reg = Regressor(model, bar_dist, get_default_device())
    reg.fit(X_train, y_train)
    # Find good range for y
    y_explore = np.linspace(-10.0, 10.0, steps)
    nodelist = [v for v in values.keys() if v != 'y']
    X_test = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
    log_p_explore = reg.log_ppd(X_test, y_explore, **kwargs)
    p_explore = np.exp(log_p_explore)
    mask = p_explore > EPS
    indices = np.where(mask)[0]
    start_idx = max(0, indices[0])
    end_idx = min(len(y_explore) - 1, indices[-1])
    a = y_explore[start_idx]
    b = y_explore[end_idx]
    # Plot and change range
    y = np.linspace(a, b, steps)
    log_probs = reg.log_ppd(X_test, y, **kwargs)
    probs = np.exp(log_probs)
    ax.plot(y, probs, **style)
    curr_min, curr_max = ax.get_xlim()
    ax.set_xlim(min(curr_min, a), max(curr_max, b))


def mcmc_suite(generator: torch.Generator, output_dir: str, include_mcmc: bool = True, include_pfn: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample the training data D = (X, y)
    seed = int(torch.randint(0, 10000, (1,), generator=generator).item())
    sample_shape = (5,)
    test_sample_shape = (1,)
    
    prior = ObservationalDataLoader(1, 1, fixed_graph=True, seed=seed)
    data = next(iter(prior))
    scm = data['graph_information']['scm']
    graph = data['graph_information']['graph']
    scm.sample_noise(sample_shape, generator=generator)
    values = scm.propagate()
    plt.figure()
    plot_graph(graph, f"{output_dir}/true_graph.png", **DRAWING_STYLE)
    scm.sample_noise(test_sample_shape, generator=generator)
    test_sample = scm.propagate()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(x=test_sample['y'].item(), color='black', linestyle='--', linewidth=1, label="True Value")
    
    if include_mcmc:
        # Perform MCMC
        prior = ObservationalDataLoader(100, 1, fixed_graph=True, seed=seed+1)
        chain = mcmc(values, prior, generator)
        # Evaluate PPD on test sample
        style = {'label': 'ppd', 'color': 'blue', 'linestyle': '-'}
        plot_ppd(ax, test_sample, [scm for scm, _ in chain], style=style)
    
    if include_pfn:
        nodelist = [v for v in values.keys() if v != 'y']
        X_train = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
        y_train = values['y'].cpu().numpy()
        model_names = ["basic_5", "basic_5_pe"]
        model_types = {"basic_5": "pfn", "basic_5_pe": "pos_encoding"}
        model_colors = {"basic_5": "red", "basic_5_pe": "violet"}
        for model_name in model_names:
            model_path = f"workdir/{model_name}"
            style = {'label': model_name, 'color': model_colors[model_name], 'linestyle': '--'}
            plot_ppd_pfn(ax, test_sample, X_train, y_train, model_path, model_types[model_name], style=style, adjacency_matrix=data['graph_information']['adjacency_matrix'])
        
    ax.set_xlabel("y")
    ax.set_ylabel("p(y)")
    ax.set_title("PPD Comparison")
    ax.legend()
    ax.grid(True)
    fig.savefig(f"{output_dir}/ppds.png", dpi=300)


if __name__ == "__main__":
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"evaluation/output/{datetime_str}"
    
    seed = 43
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    for i in range(20):
        mcmc_suite(generator, f"{output_dir}/run_{i}", include_mcmc=True, include_pfn=True)