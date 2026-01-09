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


def likelihood(values: Dict, test_sample: Dict, scm: SCM) -> float:
    log_prob = 0.0
    shape_values = values[list(values.keys())[0]].shape
    for idx in np.ndindex(shape_values):
        values_i = {v: values[v][idx] for v in values}
        log_prob += scm.total_log_probability(values_i)
    values_x = {v: test_sample[v] for v in test_sample if v != 'y'}
    log_prob += scm.marginal(values_x)
    return log_prob


def cheap_likelihood(values: Dict, test_sample: Dict, scm: SCM) -> float:
    # ignore the conditioning on the new test sample x'
    log_prob = 0.0
    shape_values = values[list(values.keys())[0]].shape
    for idx in np.ndindex(shape_values):
        values_i = {v: values[v][idx] for v in values}
        log_prob += scm.total_log_probability(values_i)
    return log_prob


def ignore_context(values: Dict, test_sample: Dict, scm: SCM) -> float:
    values_x = {v: test_sample[v] for v in test_sample if v != 'y'}
    return scm.marginal(values_x)


@torch.no_grad()
def mcmc(values: Dict, test_sample: Dict, prior: ObservationalDataLoader, generator: torch.Generator, likelihood_fn=likelihood) -> List[Tuple[SCM, float]]:
    """Perform basic MCMC where the proposal distribution is just the prior."""
    prior_iter = iter(prior)
    incumbent = next(prior_iter)['graph_information']['scm']
    incumbent_log_prob = likelihood_fn(values, test_sample, incumbent)
    # keep track of triples: (scm, log_prob, weight)
    chain = [(incumbent, incumbent_log_prob, 1)]
    for i, data in enumerate(prior):
        print(f"MCMC step {i+1}/{len(prior)}...")
        proposal: SCM = data['graph_information']['scm']
        proposal_log_prob = likelihood_fn(values, test_sample, proposal)
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
            chain.append((proposal, proposal_log_prob, 1))
            incumbent = proposal
            incumbent_log_prob = proposal_log_prob
        else:
            print("Move rejected...")
            chain[-1] = (incumbent, incumbent_log_prob, chain[-1][2] + 1)
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


def plot_ppd(ax, values: Dict, samples: List, style: Dict, steps: int = 200):
    # Find good range for y
    y_explore = torch.linspace(-10.0, 10.0, steps)
    likelihoods_explore = [torch.exp(scm.log_likelihood_batch(values, y_explore)) for scm, _, _ in samples]
    weights = [w for _, _, w in samples]
    p_explore = sum(t * w for t, w in zip(likelihoods_explore, weights)) / sum(weights)
    mask = p_explore > EPS
    indices = torch.where(mask)[0]
    buffer = 1
    start_idx = max(0, indices[0] - buffer)
    end_idx = min(len(y_explore) - 1, indices[-1] + buffer)
    a = y_explore[start_idx]
    b = y_explore[end_idx]
    # Plot and change range
    y = torch.linspace(a, b, steps)
    likelihoods = [torch.exp(scm.log_likelihood_batch(values, y)) for scm, _, _ in samples]
    for ll in likelihoods:
        ax.plot(y, ll, **{k: v for k, v in style.items() if k != 'label'}, alpha=0.02)
    probs = sum(t * w for t, w in zip(likelihoods, weights)) / sum(weights)
    ax.plot(y, probs, **style)
    curr_min, curr_max = ax.get_xlim()
    ax.set_xlim(min(curr_min, a), max(curr_max, b))
    

def plot_ppd_pfn(ax, values: Dict, X_train, y_train, model_path: str, model_type: str, style: Dict, steps: int = 200, **kwargs):
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
    buffer = 1
    start_idx = max(0, indices[0] - buffer)
    end_idx = min(len(y_explore) - 1, indices[-1] + buffer)
    a = y_explore[start_idx]
    b = y_explore[end_idx]
    # Plot and change range
    y = np.linspace(a, b, steps)
    log_probs = reg.log_ppd(X_test, y, **kwargs)
    # pred = reg.predict(X_test, **kwargs)
    # ax.axvline(x=pred.item(), color=style.get('color', 'black'), linestyle='--')
    probs = np.exp(log_probs)
    ax.plot(y, probs, **style)
    curr_min, curr_max = ax.get_xlim()
    ax.set_xlim(min(curr_min, a), max(curr_max, b))


def mcmc_suite(generator: torch.Generator, output_dir: str, include_mcmc: bool = True, include_pfn: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample the training data D = (X, y)
    seed = int(torch.randint(0, 10000, (1,), generator=generator).item())
    sample_shape = (10,)
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
        prior = ObservationalDataLoader(200, 1, fixed_graph=True, seed=seed+1)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        style = {'label': 'PPD: p(y|x, D, graph)', 'color': 'blue', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        # MCMC ignoring context
        prior = ObservationalDataLoader(200, 1, fixed_graph=True, seed=seed+2)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=ignore_context)
        style = {'label': 'p(y|x, graph)', 'color': 'cyan', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
    
    if include_pfn:
        nodelist = [v for v in values.keys() if v != 'y']
        X_train = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
        y_train = values['y'].cpu().numpy()
        model_names = ["basic_5", "basic_5_pe", "basic_5_graph"]
        model_types = {"basic_5": "pfn", "basic_5_pe": "pos_encoding", "basic_5_graph": "binary"}
        model_colors = {"basic_5": "red", "basic_5_pe": "violet", "basic_5_graph": "orange"}
        model_labels = {"basic_5": "PFN (baseline, no graph info)", "basic_5_pe": "PFN (trained on fixed graph)", "basic_5_graph": "PFN (incorporating graph info)"}
        model_names = ["ppd_01_09_17_31", "ppd_graph_01_09_17_33", "ppd_pe_01_09_17_36"]
        model_types = {"ppd_01_09_17_31": "pfn", "ppd_graph_01_09_17_33": "binary", "ppd_pe_01_09_17_36": "pos_encoding"}
        model_colors = {"ppd_01_09_17_31": "red", "ppd_graph_01_09_17_33": "orange", "ppd_pe_01_09_17_36": "violet"}
        model_labels = {"ppd_01_09_17_31": "PFN (baseline, no graph info)", "ppd_graph_01_09_17_33": "PFN (incorporating graph info)", "ppd_pe_01_09_17_36": "PFN (trained on fixed graph)"}
        for model_name in model_names:
            model_path = f"workdir/{model_name}"
            style = {'label': model_labels[model_name], 'color': model_colors[model_name], 'linestyle': '--'}
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
    
    seed = 41
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    for i in range(20):
        mcmc_suite(generator, f"{output_dir}/run_{i}", include_mcmc=True, include_pfn=True)