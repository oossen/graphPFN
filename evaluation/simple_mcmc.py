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
from priors.basic_dataloader_graph_prior import ObservationalDataLoader
from configs.likelihood_training_configs import prior_config
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
    y_explore = torch.linspace(-10.0, 10.0, steps)
    likelihoods_explore = [torch.exp(scm.log_likelihood_batch(values, y_explore)) for scm, _, _ in samples]
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
    likelihoods = [torch.exp(scm.log_likelihood_batch(values, y)) for scm, _, _ in samples]
    # for ll in likelihoods:
    #    ax.plot(y, ll, **{k: v for k, v in style.items() if k != 'label'}, alpha=0.02)
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
    sample_shape = (prior_config['dataset_config']['number_train_samples_per_dataset']['value'],)
    test_sample_shape = (1,)
    
    prior = ObservationalDataLoader(100, 1, prior_config, seed=seed)
    prior_config['dataset_config']['number_test_samples_per_dataset'] = {'value': 1}
    
    data = next(iter(prior))
    scm = data['graph_information']['scm']
    graph = data['graph_information']['graph']
    scm.sample_noise(sample_shape, generator=generator)
    values = scm.propagate()
    plt.figure()
    plot_graph(graph, f"{output_dir}/true_graph.png", **DRAWING_STYLE)
    scm.sample_noise(test_sample_shape, generator=generator)
    test_sample = scm.propagate()
    test_sample = {v: value.item() for v, value in test_sample.items()}
    
    # Save y and normalized ys to file
    ys = values['y'].cpu().numpy()
    ys_mean, ys_std = np.mean(ys), np.std(ys) + 1e-8
    ys_n = (ys - ys_mean) / ys_std
    with open(f"{output_dir}/values.txt", "w") as f:
        f.write(f"training y values: {ys}\n")
        f.write(f"mu, sigma: {ys_mean}, {ys_std}\n")
        f.write(f"normalized training y values: {ys_n}\n")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(x=test_sample['y'], color='black', linestyle='--', linewidth=1, label="True Value")
    
    if include_mcmc:
        # MCMC with graph prior
        prior = ObservationalDataLoader(2000, 1, prior_config, seed=seed+1).make_iter(data['graph_information']['adjacency_matrix'])
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'p(y|D, graph)', 'color': 'blue', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        # not ignoring info from test sample
        prior = ObservationalDataLoader(2000, 1, prior_config, seed=seed+2).make_iter(data['graph_information']['adjacency_matrix'])
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'p(y|D, x, graph)', 'color': 'cyan', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        # MCMC over entire prior
        prior = ObservationalDataLoader(20000, 1, prior_config, seed=seed+3)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'p(y|D)', 'color': 'green', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        # MCMC over entire prior again, to see if they're the same
        prior = ObservationalDataLoader(20000, 1, prior_config, seed=seed+4)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'p(y|D)', 'color': 'green', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        # MCMC over entire prior not ignoring info from test sample
        prior = ObservationalDataLoader(20000, 1, prior_config, seed=seed+5)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'p(y|x, D)', 'color': 'violet', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        # true SCM
        chain = [(scm, 0.0, 1)]
        style = {'label': 'p(y|x, true SCM)', 'color': 'black', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
    
    if include_pfn:
        nodelist = [v for v in values.keys() if v != 'y']
        X_train = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
        y_train = values['y'].cpu().numpy()
        model_names = ["likelihood_training_01_26_15_45", "likelihood_training_01_26_15_56"]
        model_types = {"likelihood_training_01_26_15_45": "pfn", "likelihood_training_01_26_15_56": "pfn"}
        model_colors = {"likelihood_training_01_26_15_45": "red", "likelihood_training_01_26_15_56": "orange"}
        model_labels = {"likelihood_training_01_26_15_45": "p(y|x, D)", "likelihood_training_01_26_15_56": "p(y|x, D) (CE)"}
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
    
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    for i in range(20):
        mcmc_suite(generator, f"{output_dir}/run_{i}", include_mcmc=True, include_pfn=True)