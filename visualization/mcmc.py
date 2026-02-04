import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from datetime import datetime
from pfns.bar_distribution import FullSupportBarDistribution
from dopfnprior.scm.scm import SCM
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.observational_dataloader import ObservationalDataLoader
from configs.default_configs import prior_config, training_config
from visualization.plotting import plot_graph
from tfmplayground.utils import get_default_device


EPS = 1e-2

FIXED_POS = {
    'x0': (0, 0),
    'x1': (1, 0),
    'x2': (2, 1),
    'x3': (2, 2),
    'x4': (1, 3),
    'x5': (0, 3),
    'x6': (-1, 2),
    'y': (-1, 1),
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
    log_probs = scm.total_log_probability(values)
    log_prob = log_probs.sum().item()
    values_x = {v: test_sample[v] for v in test_sample if v != 'y'}
    log_prob += scm.marginal(values_x, steps=100, low=-10.0, high=10.0)
    return log_prob


def cheap_likelihood(values: Dict, test_sample: Dict, scm: SCM) -> float:
    # ignore the conditioning on the new test sample x'
    log_probs = scm.total_log_probability(values)
    return log_probs.sum().item()


def ignore_context(values: Dict, test_sample: Dict, scm: SCM) -> float:
    values_x = {v: test_sample[v] for v in test_sample if v != 'y'}
    return scm.marginal(values_x)


@torch.no_grad()
def mcmc(values: Dict, test_sample: Dict, prior, generator: torch.Generator, likelihood_fn=likelihood) -> List[Tuple[SCM, float, int]]:
    """Perform basic MCMC where the proposal distribution is just the prior."""
    prior_iter = iter(prior)
    incumbent = next(prior_iter)['graph_information']['scm']
    incumbent_log_prob = likelihood_fn(values, test_sample, incumbent)
    # keep track of triples: (scm, log_prob, weight)
    chain = [(incumbent, incumbent_log_prob, 1)]
    for i, data in enumerate(prior_iter):
        print(f"MCMC step {i+1}...")
        proposal: SCM = data['graph_information']['scm']
        proposal_log_prob = likelihood_fn(values, test_sample, proposal)
        
        log_acceptance_ratio = proposal_log_prob - incumbent_log_prob
        log_sample = torch.rand((1,), generator=generator).log().item()
        
        if log_sample < log_acceptance_ratio:
            chain.append([proposal, proposal_log_prob, 1])
            incumbent = proposal
            incumbent_log_prob = proposal_log_prob
        else:
            chain[-1] = (incumbent, incumbent_log_prob, chain[-1][2] + 1)
    return chain


def plot_ppd(ax, values: Dict, samples: List, style: Dict, steps: int = 100):
    # Find good range for y
    values = {v: values[v].unsqueeze(0) for v in values}
    y_explore = torch.linspace(-10.0, 10.0, steps)
    likelihoods_explore = [torch.exp(scm.log_likelihood_batch(values, y_explore.unsqueeze(0)))[0][0] for scm, _, _ in samples]
    weights = [w for _, _, w in samples]
    weighted_sum = torch.stack([t * w for t, w in zip(likelihoods_explore, weights)]).sum(dim=0)
    p_explore = weighted_sum / sum(weights)
    eps = 0.01 * p_explore.max()
    mask = p_explore > eps
    indices = torch.where(mask)[0]
    buffer = 1
    start_idx = max(0, indices[0] - buffer)
    end_idx = min(len(y_explore) - 1, indices[-1] + buffer)
    a = y_explore[start_idx]
    b = y_explore[end_idx]
    # Plot and change range
    y = torch.linspace(a, b, steps)
    likelihoods = [torch.exp(scm.log_likelihood_batch(values, y.unsqueeze(0)))[0][0] for scm, _, _ in samples]
    # for ll in likelihoods:
    #    ax.plot(y, ll, **{k: v for k, v in style.items() if k != 'label'}, alpha=0.02)
    probs = sum(t * w for t, w in zip(likelihoods, weights)) / sum(weights)
    ax.plot(y, probs, **style)
    curr_min, curr_max = ax.get_xlim()
    ax.set_xlim(min(curr_min, a), max(curr_max, b))
    

def plot_ppd_pfn(ax, values: Dict, X_train, y_train, model_path: str, style: Dict, steps: int = 200, **kwargs):
    # Initialize and fit model
    model = init_model_from_state_dict_file(f"{model_path}/latest_checkpoint.pth")
    buckets = training_config['buckets']
    reg = Regressor(model, buckets)
    reg.fit(X_train, y_train)
    # Find good range for y
    y_explore = np.linspace(-10.0, 10.0, steps)
    nodelist = [v for v in values.keys() if v != 'y']
    X_test = np.array([values[v] for v in nodelist])
    X_test = X_test.reshape(1, -1)
    log_p_explore = reg.log_ppd(X_test, y_explore, **kwargs)
    p_explore = np.exp(log_p_explore)
    eps = 0.01 * p_explore.max()
    mask = p_explore > eps
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
    sample_shape = (5,)
    test_sample_shape = (1,)
    
    # restrict prior to small graphs
    prior_config['graph_config']['num_nodes'] = {'value': 5}
    prior = ObservationalDataLoader(100, 1, prior_config, seed=seed)
    
    data = next(iter(prior))
    scm = data['graph_information']['scm']
    graph = data['graph_information']['graph']
    adj = data['graph_information']['adj']
    scm.sample_noise(sample_shape, generator=generator)
    values = scm.propagate()
    plt.figure()
    plot_graph(graph, f"{output_dir}/graph.png", **DRAWING_STYLE)
    scm.sample_noise(test_sample_shape, generator=generator)
    test_sample = scm.propagate()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(x=test_sample['y'], color='black', linestyle='--', linewidth=1, label="True Value")
    
    if include_mcmc:
        # MCMC with graph prior
        prior = ObservationalDataLoader(100, 1, prior_config, seed=seed+1).make_iter(adj)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'p(y|D, graph) (MCMC)', 'color': 'orange', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        # MCMC with graph prior
        prior = ObservationalDataLoader(100, 1, prior_config, seed=seed+2).make_iter(adj)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'p(y|D, graph) (MCMC)', 'color': 'orange', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        
    if include_pfn:
        nodelist = [v for v in values.keys() if v != 'y']
        X_train = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
        y_train = values['y'].cpu().numpy()
        model_names = ["baseline_02_03_17_45", "baseline_nll_02_03_17_45", "binary_attention_02_03_17_46"]
        model_colors = {"baseline_02_03_17_45": "red", "baseline_nll_02_03_17_45": "green", "binary_attention_02_03_17_46": "orange"}
        model_labels = {"baseline_02_03_17_45": "p(y|x, D) (PFN)", "baseline_nll_02_03_17_45": "p(y|x, D) (PFN-NLL)", "binary_attention_02_03_17_46": "p(y|x, D, graph) (PFN-Att)"}
        for model_name in model_names:
            model_path = f"workdir/{model_name}"
            style = {'label': model_labels[model_name], 'color': model_colors[model_name], 'linestyle': '--'}
            plot_ppd_pfn(ax, test_sample, X_train, y_train, model_path, style=style, **data['graph_information'])
        
    ax.set_xlabel("y")
    ax.set_ylabel("p(y)")
    ax.set_title("PPD Comparison")
    ax.legend()
    ax.grid(True)
    fig.savefig(f"{output_dir}/ppds.png", dpi=300)


if __name__ == "__main__":
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    
    seed = 43
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    for i in range(50):
        mcmc_suite(generator, f"{output_dir}/run_{i}", include_mcmc=True, include_pfn=True)