import math
import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from datetime import datetime
from collections import Counter
from collections import defaultdict
import networkx as nx
from copy import deepcopy
from dopfnprior.scm.scm import SCM
from dopfnprior.scm.simple_mechanism import SimpleMechanism
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.observational_dataloader import ObservationalDataLoader
from configs.simple_configs import prior_config, training_config
from visualization.plotting import plot_graph
import torch.distributions as dist
from dopfnprior.utils.sampling import TorchDistributionSampler


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
    log_prob += scm.marginal(values_x, steps=100, low=-10.0, high=10.0).item()
    return log_prob


def cheap_likelihood(values: Dict, test_sample: Dict, scm: SCM) -> float:
    # ignore the conditioning on the new test sample x'
    log_probs = scm.total_log_probability(values)
    return log_probs.sum().item()


def evidence(values: Dict, test_sample: Dict, graph: nx.DiGraph, prior, likelihood_fn=likelihood) -> float:
    """Estimate p(x, D | graph) by sampling from the prior."""
    log_evidence = float('-inf')
    for data in prior.make_iter(nx.to_numpy_array(graph)):
        scm = data['graph_information']['scm']
        log_prob = likelihood_fn(values, test_sample, scm)
        log_evidence = torch.logaddexp(torch.tensor(log_evidence), torch.tensor(log_prob)).item()
    log_evidence -= math.log(len(prior))
    return log_evidence


def perturbate(incumbent_scm: SCM, generator: torch.Generator) -> SCM:
    selector = torch.rand((1,), generator=generator)
    if selector < 0.5:
        # perturb a mechanism
        nodes = list(incumbent_scm.dag.nodes())
        new_mechanisms = {v: incumbent_scm.mechanisms[v] for v in nodes}
        node_idx = int(torch.randint(0, len(nodes), (1,), generator=generator).item())
        changed_v = nodes[node_idx]
        new_mech: SimpleMechanism = deepcopy(new_mechanisms[changed_v])
        # perturb bias
        incumbent = new_mech.bias.item()
        proposal = normal_proposal(incumbent, -1.0, 1.0, 0.5, generator)
        new_mech.bias.fill_(proposal)
        parents = list(incumbent_scm.dag.predecessors(changed_v))
        for w in parents:
            incumbent = new_mech.weights[w].item()
            proposal = normal_proposal(incumbent, -1.0, 1.0, 0.5, generator)
            new_mech.weights[w].fill_(proposal)
        print("Perturbed mechanisms...")
        new_mechanisms[changed_v] = new_mech
        new_scm = SCM(incumbent_scm.dag, new_mechanisms, incumbent_scm.noise)
        return new_scm
    else:
        nodes = list(incumbent_scm.dag.nodes())
        node_idx = int(torch.randint(0, len(nodes), (1,), generator=generator).item())
        changed_v = nodes[node_idx]
        new_noise = {v: incumbent_scm.noise[v] for v in nodes}
        incumbent = incumbent_scm.noise[changed_v].std()
        proposal = normal_proposal(incumbent, 0.1, 20.0, 0.5, generator)
        new_noise[changed_v] = TorchDistributionSampler(dist.Normal(loc=0.0, scale=proposal))
        print("Perturbed noise...")
        new_scm = SCM(incumbent_scm.dag, incumbent_scm.mechanisms, new_noise)
        return new_scm


def uniform_proposal(incumbent: float, low: float, high: float, max_change: float, generator: torch.Generator) -> float:
    change = torch.rand((1,), generator=generator).item() * 2 * max_change - max_change
    proposal = incumbent + change
    if proposal >= low and proposal < high:
        return proposal
    else:
        return incumbent
    

def normal_proposal(incumbent: float, low: float, high: float, std: float, generator: torch.Generator) -> float:
    change = torch.randn((1,), generator=generator).item() * std
    proposal = incumbent + change
    while proposal < low or proposal > high:
        if proposal < low:
            # Distance below low is reflected back up
            proposal = 2 * low - proposal
        elif proposal > high:
            # Distance above high is reflected back down
            proposal = 2 * high - proposal
            
    return proposal


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
            chain.append((proposal, proposal_log_prob, 1))
            incumbent = proposal
            incumbent_log_prob = proposal_log_prob
        else:
            chain[-1] = (incumbent, incumbent_log_prob, chain[-1][2] + 1)
    return chain


@torch.no_grad()
def fancy_mcmc(values: Dict, 
               test_sample: Dict, 
               prior,
               initial_scm: SCM, 
               steps: int, 
               burn_in: int,
               thinning: int,
               generator: torch.Generator, 
               likelihood_fn=likelihood) -> List[Tuple[SCM, float, int]]:
    """Perform MCMC with a symmetric proposal distribution."""
    incumbent = initial_scm
    incumbent_log_prob = likelihood_fn(values, test_sample, incumbent)
    incumbent_prior_log_prob = prior.log_likelihood(incumbent)
    # keep track of triples: (scm, log_prob, weight)
    chain = [(incumbent, (incumbent_log_prob, incumbent_prior_log_prob), 1)]
    for i in range(steps):
        print(f"MCMC step {i+1}...")
        incumbent = chain[-1][0]
        proposal: SCM = perturbate(incumbent, generator)
        proposal_log_prob = likelihood_fn(values, test_sample, proposal)
        proposal_prior_log_prob = prior.log_likelihood(proposal)
        incumbent_log_prob, incumbent_prior_log_prob = chain[-1][1]
        log_acceptance_ratio = proposal_log_prob + proposal_prior_log_prob - incumbent_log_prob - incumbent_prior_log_prob
        print(f"Perturbation, log likelihoods: {incumbent_log_prob} -> {proposal_log_prob}, log priors: {incumbent_prior_log_prob} -> {proposal_prior_log_prob}")
        
        log_sample = torch.rand((1,), generator=generator).log().item()
        
        if log_sample < log_acceptance_ratio:
            chain.append((proposal, (proposal_log_prob, proposal_prior_log_prob), 1))
            print("Accepted!")
        else:
            chain[-1] = (chain[-1][0], chain[-1][1], chain[-1][2] + 1)
            print("Rejected!")
    return process_mcmc_rle(chain, burn_in=burn_in, k=thinning)


def process_mcmc_rle(sequence, burn_in=0, k=1):
    """Applies burn-in and thinning to a run-length encoded MCMC sequence."""
    result = []
    current_idx = 0  # absolute index in the original uncompressed sequence
    target_idx = burn_in # the index of the next sample we want to keep
    for value, prob, count in sequence:
        run_end = current_idx + count
        if target_idx < run_end:
            num_hits = (run_end - 1 - target_idx) // k + 1
            result.append((value, prob, num_hits))
            target_idx = target_idx + (num_hits * k)
        current_idx = run_end
    return result


def plot_ppd(ax, values: Dict, samples: List, style: Dict, steps: int = 100):
    # Find good range for y
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
    

def visualize_chain(chain: List[Tuple[SCM, float, int]], true_scm: SCM, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # graph
    nodes = list(chain[0][0].dag.nodes())
    average_graph = nx.DiGraph()
    average_graph.add_nodes_from(nodes)
    edge_counts = Counter()
    for scm, _, weight in chain:
        edge_counts.update({(u, v): weight for u, v in scm.dag.edges()})
    num_samples = sum(weight for _, _, weight in chain)
    for (u, v), count in edge_counts.items():
        probability = count / num_samples
        average_graph.add_edge(u, v, weight=probability)
    plt.figure()
    plot_graph(average_graph, f"{output_dir}/average_graph.png", **DRAWING_STYLE)
    
    # weights
    weights = defaultdict(list)
    biases = defaultdict(list)
    true_weights = {(u, v): true_scm.mechanisms[u].weights[v].item() for u in nodes for v in nodes}
    true_biases = {u: true_scm.mechanisms[u].bias.item() for u in nodes}
    for scm, _, weight in chain:
        for u in nodes:
            biases[u].append((scm.mechanisms[u].bias.item(), weight))
            for v in nodes:
                weights[(u, v)].append((scm.mechanisms[u].weights[v].item(), weight))
                
    n_plots = len(weights) + len(biases)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axes_flat = axes.flatten()
    for i, (key, pairs) in enumerate(weights.items()):
        ax = axes_flat[i]
        p_vals, w_vals = zip(*pairs)
        ax.hist(p_vals, weights=w_vals, bins=50, density=True, 
                alpha=0.7, color='skyblue', edgecolor='black')
        line_x = true_weights[key]
        ax.axvline(x=line_x, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"{key[1]} -> {key[0]}")
    for i, (key, pairs) in enumerate(biases.items(), start=len(weights)):
        ax = axes_flat[i]
        p_vals, w_vals = zip(*pairs)
        ax.hist(p_vals, weights=w_vals, bins=50, density=True, 
                alpha=0.7, color='lightgreen', edgecolor='black')
        line_x = true_biases[key]
        ax.axvline(x=line_x, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"bias {key}")
    plt.savefig(f"{output_dir}/weights.png", dpi=300)
    
    # noise
    noises = defaultdict(list)
    true_noises = {u: true_scm.noise[u].std() for u in nodes}
    for scm, _, weight in chain:
        for u in nodes:
            noises[u].append((scm.noise[u].std(), weight))
    fig, axes = plt.subplots(nrows=1, ncols=len(noises), figsize=(5 * len(noises), 4), constrained_layout=True)
    axes_flat = axes.flatten()
    for i, (key, pairs) in enumerate(noises.items()):
        ax = axes_flat[i]
        p_vals, w_vals = zip(*pairs)
        ax.hist(p_vals, weights=w_vals, bins=50, density=True, 
                alpha=0.7, color='salmon', edgecolor='black')
        line_x = true_noises[key]
        ax.axvline(x=line_x, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"noise std {key}")
    plt.savefig(f"{output_dir}/noises.png", dpi=300)
    

def mcmc_suite(prior, generator: torch.Generator, output_dir: str, include_mcmc: bool = True, include_pfn: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample the training data D = (X, y)
    sample_shape = (1, 2) # 1 batch, 5 samples
    test_sample_shape = (1, 1) # 1 batch, 1 sample
    data = next(iter(prior))
    scm = data['graph_information']['scm']
    graph = data['graph_information']['graph']
    scm.sample_noise(sample_shape, generator=generator)
    values = scm.propagate()
    scm.sample_noise(test_sample_shape, generator=generator)
    test_sample = scm.propagate()
    
    plt.figure()
    plot_graph(graph, f"{output_dir}/graph.png", **DRAWING_STYLE)
    
    true_graph_tuple = (tuple(graph.nodes()), tuple(sorted(graph.edges())))
    print(f"True graph: {true_graph_tuple}")
    graph_posteriors = {}
    for graph_tuple in prior.graph_counts:
        g = nx.DiGraph()
        g.add_nodes_from(graph_tuple[0])
        g.add_edges_from(graph_tuple[1])
        log_prob = evidence(values, test_sample, g, prior)
        prior_prob = math.log(prior.graph_counts.get(graph_tuple, 1))
        print(f"Graph {graph_tuple}:")
        print(f"Evidence log: {log_prob}, prior prob: {prior_prob}")
        graph_posteriors[graph_tuple] = log_prob + prior_prob
    # compute softmax to get actual probabilities
    max_log_posterior = max(graph_posteriors.values())
    shifted_exp = {graph_tuple: math.exp(prob - max_log_posterior) for graph_tuple, prob in graph_posteriors.items()}
    total_sum = sum(shifted_exp.values())
    probs = {graph_tuple: prob / total_sum for graph_tuple, prob in shifted_exp.items()}
    print(probs.items())
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(x=test_sample['y'], color='black', linestyle='--', linewidth=1, label="True Value")
    
    if include_mcmc:
        steps, burn_in, thinning = 5000, 1000, 2
        distributions = {}
        for graph_tuple in prior.graph_counts:
            if probs[graph_tuple] < 1e-2 and graph_tuple != true_graph_tuple:
                continue
            g = nx.DiGraph()
            g.add_nodes_from(graph_tuple[0])
            g.add_edges_from(graph_tuple[1])
            prior_iter = prior.make_iter(nx.to_numpy_array(g))
            initial_scm = next(prior_iter)['graph_information']['scm']
            chain = fancy_mcmc(values, test_sample, prior, initial_scm, steps, burn_in, thinning, generator)
            visualize_chain(chain, scm, f"{output_dir}/mcmc_graph_{graph_tuple}")
            y = torch.linspace(-10.0, 10.0, steps)
            likelihoods = [torch.exp(scm.log_likelihood_batch(test_sample, y.unsqueeze(0)))[0][0] for scm, _, _ in chain]
            weights = [w for _, _, w in chain]
            weighted_sum = torch.stack([t * w for t, w in zip(likelihoods, weights)]).sum(dim=0)
            distributions[graph_tuple] = weighted_sum / sum(weights)
            # style = {'color': 'orange', 'linestyle': '-', 'alpha': 0.1}
            # ax.plot(y, distributions[graph_tuple], **style)
        style = {'label': 'p(y|x, D, γ) (MCMC)', 'color': 'orange', 'linestyle': '-', 'alpha': 1.0}
        ax.plot(y, distributions[true_graph_tuple], **style)
        weighted_sum = sum(distributions[graph_tuple] * probs[graph_tuple] for graph_tuple in distributions) / sum(probs.values())
        style = {'label': 'p(y|x, D) (MCMC)', 'color': 'red', 'linestyle': '-', 'alpha': 1.0}
        ax.plot(y, weighted_sum, **style)
        eps = 0.01 * weighted_sum.max()
        mask = (weighted_sum > eps) | (distributions[true_graph_tuple] > eps)
        indices = torch.where(mask)[0]
        buffer = 1
        start_idx = max(0, indices[0] - buffer)
        end_idx = min(len(y) - 1, indices[-1] + buffer)
        a = y[start_idx].item()
        b = y[end_idx].item()
        ax.set_xlim(a, b)
        
    if include_pfn:
        nodelist = [v for v in values.keys() if v != 'y']
        X_train = torch.stack([values[v][0] for v in nodelist], dim=-1).cpu().numpy()
        y_train = values['y'][0].cpu().numpy()
        model_names = ["simple_binary_attention_fallback_02_14_22_21","simple_02_14_22_20"]
        model_colors = {"simple_binary_attention_fallback_02_14_22_21": "orange", "simple_02_14_22_20": "red"}
        model_labels = {"simple_binary_attention_fallback_02_14_22_21": "p(y|x, D, γ) (PFN)", "simple_02_14_22_20": "p(y|x, D) (PFN)"}
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
    
    seed = 100
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior_config['graph_config']['num_nodes'] = {'value': 3}
    prior = ObservationalDataLoader(100, 1, prior_config, seed=seed)
    prior._make_statistics(steps=10000)
    
    for i in range(20):
        mcmc_suite(prior, generator, f"{output_dir}/run_{i}", include_mcmc=True, include_pfn=True)