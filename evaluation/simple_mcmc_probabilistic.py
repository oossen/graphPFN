import math
import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from datetime import datetime
from pfns.bar_distribution import FullSupportBarDistribution
from dopfnprior.scm.scm import SCM
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.basic_dataloader_probabilistic import ObservationalDataLoader
from configs.probabilistic_configs import prior_config
from visualization.plotting import plot_graph
from tfmplayground.utils import get_default_device


EPS = 1e-2

FIXED_POS = {
    'x0': (1, 3),
    'x1': (3, 3),
    'x2': (1, 1),
    'y': (3, 1)
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


def likelihood(values: Dict, test_sample: Dict, theta: Dict) -> float:
    log_prob = 0.0
    shape_values = values[list(values.keys())[0]].shape
    scm_test = theta['graph_information']['scm_test']
    scm_train = theta['graph_information']['scm_train']
    for idx in np.ndindex(shape_values):
        values_i = {v: values[v][idx] for v in values}
        log_prob += scm_train.total_log_probability(values_i)
    values_x = {v: test_sample[v] for v in test_sample if v != 'y'}
    log_prob += scm_test.marginal(values_x)
    return log_prob


def cheap_likelihood(values: Dict, test_sample: Dict, theta: Dict) -> float:
    # ignore the conditioning on the new test sample x'
    log_prob = 0.0
    shape_values = values[list(values.keys())[0]].shape
    scm_train = theta['graph_information']['scm_train']
    for idx in np.ndindex(shape_values):
        values_i = {v: values[v][idx] for v in values}
        log_prob += scm_train.total_log_probability(values_i)
    return log_prob


def ignore_context(values: Dict, test_sample: Dict, theta: Dict) -> float:
    scm_test = theta['graph_information']['scm_test']
    values_x = {v: test_sample[v] for v in test_sample if v != 'y'}
    return scm_test.marginal(values_x)


@torch.no_grad()
def mcmc(values: Dict, test_sample: Dict, prior, generator: torch.Generator, likelihood_fn=likelihood) -> List[Tuple[SCM, float, int]]:
    """Perform basic MCMC where the proposal distribution is just the prior."""
    prior_iter = iter(prior)
    incumbent = next(prior_iter)
    incumbent_log_prob = likelihood_fn(values, test_sample, incumbent)
    # keep track of triples: (scm, log_prob, weight)
    chain = [(incumbent, incumbent_log_prob, 1)]
    for i, data in enumerate(prior):
        print(f"MCMC step {i+1}...")
        proposal = data
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


def plot_ppd(ax, values: Dict, samples: List, style: Dict, steps: int = 200):
    scms = [theta['graph_information']['scm_test'] for theta, _, _ in samples]
    # Find good range for y
    y_explore = torch.linspace(-10.0, 10.0, steps)
    likelihoods_explore = [torch.exp(scm.log_likelihood_batch(values, y_explore)) for scm in scms]
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
    likelihoods = [torch.exp(scm.log_likelihood_batch(values, y)) for scm in scms]
    # for ll in likelihoods:
    #    ax.plot(y, ll, **{k: v for k, v in style.items() if k != 'label'}, alpha=0.02)
    probs = sum(t * w for t, w in zip(likelihoods, weights)) / sum(weights)
    ax.plot(y, probs, **style)
    curr_min, curr_max = ax.get_xlim()
    ax.set_xlim(min(curr_min, a), max(curr_max, b))
    

def plot_ppd_animated(ax, values: Dict, samples: List, style: Dict, steps: int = 200, interval: int = 1000):
    """
    Plots an animation of the PPD where each frame adds one sample to the weighted average.
    Returns a FuncAnimation object which MUST be stored in a variable to run.
    """
    scms = [theta['graph_information']['scm_test'] for theta, _, _ in samples]
    weights = torch.tensor([w for _, _, w in samples])
    
    # 1. Determine the global range (using all samples) to keep the x-axis stable
    y_explore = torch.linspace(-10.0, 10.0, steps)
    all_likelihoods_explore = torch.stack([
        torch.exp(scm.log_likelihood_batch(values, y_explore)) for scm in scms
    ])
    
    total_weighted_sum = (all_likelihoods_explore * weights.view(-1, 1)).sum(dim=0)
    p_total = total_weighted_sum / weights.sum()
    
    eps = 0.01 * p_total.max()
    indices = torch.where(p_total > eps)[0]
    buffer = 1
    start_idx = max(0, indices[0] - buffer)
    end_idx = min(len(y_explore) - 1, indices[-1] + buffer)
    a, b = y_explore[start_idx], y_explore[end_idx]

    # 2. Prepare the data for the final grid
    y = torch.linspace(a, b, steps)
    # Pre-calculate likelihoods for all samples on this grid for efficiency
    all_l_curves = torch.stack([
        torch.exp(scm.log_likelihood_batch(values, y)) for scm in scms
    ])
    
    # 3. Setup the plot object
    line, = ax.plot([], [], **style)
    
    # Update x-axis limits based on the final expected distribution
    curr_min, curr_max = ax.get_xlim()
    ax.set_xlim(min(curr_min, a.item()), max(curr_max, b.item()))
    
    # Optional: adjust y-limit dynamically or pre-set it
    ax.set_ylim(0, p_total.max().item() * 1.1)

    # 4. Define the animation update function
    def update(frame):
        # Calculate weighted average for samples up to 'frame'
        # frame goes from 0 to len(samples) - 1
        current_weights = weights[:frame + 1]
        current_curves = all_l_curves[:frame + 1]
        
        weighted_sum = (current_curves * current_weights.view(-1, 1)).sum(dim=0)
        probs = weighted_sum / current_weights.sum()
        
        line.set_data(y.numpy(), probs.numpy())
        return line,

    # 5. Create the animation
    anim = FuncAnimation(
        ax.figure, 
        update, 
        frames=len(samples), 
        interval=interval, 
        blit=True, 
        repeat=False
    )
    
    return anim
    

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
    
    prior = ObservationalDataLoader(1, 1, prior_config, seed)
    data = next(iter(prior))
    values = data['values_train']
    test_sample = data['values_test']
    graph_test = data['graph_information']['graph_test']
    graph_train = data['graph_information']['graph_train']
    plt.figure()
    plot_graph(graph_test, f"{output_dir}/graph_test.png", **DRAWING_STYLE)
    plt.figure()
    plot_graph(graph_train, f"{output_dir}/graph_train.png", **DRAWING_STYLE)
    
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
    ax.axvline(x=test_sample['y'].item(), color='black', linestyle='--', linewidth=1, label="True Value")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 5))
    ax_anim.axvline(x=test_sample['y'].item(), color='black', linestyle='--', linewidth=1, label="True Value")
    
    if include_mcmc:
        # ground truth SCM
        chain = [(data, 0.0, 1)]
        style = {'label': 'ground truth', 'color': 'cyan', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        plot_ppd(ax_anim, test_sample, chain, style=style)
        # MCMC with graph prior
        prior = ObservationalDataLoader(1000, 1, prior_config, seed+1).make_iter(graph_test)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'PPD: p(y|x, D, graph)', 'color': 'blue', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        anim = plot_ppd_animated(ax_anim, test_sample, chain, style=style)
        # MCMC with graph-agnostic prior
        prior = ObservationalDataLoader(1000, 1, prior_config, seed+2)
        chain = mcmc(values, test_sample, prior, generator, likelihood_fn=cheap_likelihood)
        print(f"Sampled {len(chain)} unique SCMS: {[(p, w) for _, p, w in chain]}")
        style = {'label': 'PPD: p(y|x, D)', 'color': 'green', 'linestyle': '-'}
        plot_ppd(ax, test_sample, chain, style=style)
        anim2 = plot_ppd_animated(ax_anim, test_sample, chain, style=style)
    
    if include_pfn:
        nodelist = [v for v in values.keys() if v != 'y']
        X_train = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
        y_train = values['y'].cpu().numpy()
        model_names = ["ppd_01_14_02_04", "ppd_pe_01_14_12_02", "ppd_pe_mixed_01_14_02_06"]
        model_types = {"ppd_01_14_02_04": "pfn", "ppd_pe_01_14_12_02": "pos_encoding", "ppd_pe_mixed_01_14_02_06": "pos_encoding"}
        model_colors = {"ppd_01_14_02_04": "red", "ppd_pe_01_14_12_02": "orange", "ppd_pe_mixed_01_14_02_06": "purple"}
        model_labels = {"ppd_01_14_02_04": "PFN (baseline, no graph info)", "ppd_pe_01_14_12_02": "PFN + Pos. Enc.", "ppd_pe_mixed_01_14_02_06": "PFN + Pos. Enc. (mixed training)"}
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
    ax_anim.set_xlabel("y")
    ax_anim.set_ylabel("p(y)")
    ax_anim.set_title("PPD Comparison (Animated)")
    ax_anim.legend()
    ax_anim.grid(True)
    anim.save(f"{output_dir}/ppds_animated.gif", writer='pillow', extra_anim=[anim2])


if __name__ == "__main__":
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"evaluation/output/{datetime_str}"
    
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    for i in range(20):
        mcmc_suite(generator, f"{output_dir}/run_{i}", include_mcmc=True, include_pfn=False)