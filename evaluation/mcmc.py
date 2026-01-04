from collections import Counter
from copy import deepcopy
import math
import os
from typing import Dict, List, Tuple
from dopfnprior.scm.scm_builder import SCMBuilder
from dopfnprior.scm.scm import SCM
from dopfnprior.utils.sampling import build_samplers, sample_parameters
from dopfnprior.causal_graph.graph_builder import GraphBuilder
from dopfnprior.utils.sampling import TorchDistributionSampler
from dopfnprior.mechanisms.simple_mechanism import SimpleMechanism
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import networkx as nx
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.prior_prob_dataloader import ObservationalDataLoader
from visualization.plotting import plot_graph
from itertools import combinations
from pfns.bar_distribution import FullSupportBarDistribution
from tfmplayground.utils import get_default_device

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
def mcmc(values: Dict,
         prior: ObservationalDataLoader, 
         initial_scm: SCM, 
         generator: torch.Generator, 
         steps: int = 200, 
         burn_in: float = 0.1, 
         step_size: int = 2,
         fixed_graph: bool = False) -> List[Tuple[SCM, float]]:
    """
    Perform the MCMC algorithm for the dataset specified by `values` with respect to `prior`.

        
    Parameters
    ----------
    values : Dict
        Dictionary containing the dataset. Keys correspond to nodes/features.
    prior : DataLoader
        The prior distribution from which we assume the data is drawn.
    initial_scm : SCM
        The first member of the MC chain
    generator : torch.Generator
        Used to make the algorithm deterministic.
    steps : int
        The number of steps the algorithm attempts to take.
    burn_in : float
        The fraction of discarded samples at the beginning of the chain.
    step_size : int
        Only every `step_size`-th sample is returned at the end.
    fixed_graph : bool
        Whether to condition on the graph underlying `initial_scm`.
    
    Returns
    -------
    chain : List[(SCM, float)]
        The list of accepted samples the algorithm has produced, paired with their posterior log probability.
    """
    incumbent = initial_scm
    incumbent_log_prob = posterior_log_prob(incumbent, values, prior)
    chain = [(incumbent, incumbent_log_prob)]
    for _ in range(steps):
        if fixed_graph:
            proposal = perturbate(incumbent, generator, perturbation_probs=(0.0, 0.5, 0.5))
        else:
            proposal = perturbate(incumbent, generator)
        proposal_log_prob = posterior_log_prob(proposal, values, prior, fixed_graph=fixed_graph)
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
            incumbent = proposal
            chain.append((proposal, proposal_log_prob))
            incumbent_log_prob = proposal_log_prob
    burn_in_index = int(len(chain) * burn_in)
    return chain[burn_in_index::step_size]


def ppd(values: Dict, samples: List[SCM], y: torch.Tensor) -> torch.Tensor:
    """
    Approximate the PPD of y given the features in `values`
    using Monte Carlo integration with the provided `samples`.
    """
    likelihoods = []
    for scm in samples:
        ll = scm.log_likelihood(values, y)
        likelihoods.append(torch.exp(ll))
    return torch.stack(likelihoods).mean(dim=0)


def plot_ppd(ax, values: Dict, samples: List[SCM], style: Dict, steps: int = 30, bounds=[-3, 3]):    
    y = torch.linspace(bounds[0], bounds[1], steps)
    probs = ppd(values, samples, y).cpu().numpy()
    ax.plot(y, probs, **style)
    

def plot_ppd_pfn(ax, values: Dict, X_train, y_train, model, style: Dict, steps: int = 200, bounds=[-3, 3], **kwargs):
    y = np.linspace(bounds[0], bounds[1], steps)
    nodelist = [v for v in values.keys() if v != 'y']
    X_test = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
    model.fit(X_train, y_train)
    log_probs = model.log_ppd(X_test, y, **kwargs)
    probs = np.exp(log_probs)
    ax.plot(y, probs, **style)


@torch.no_grad()    
def posterior_log_prob(scm: SCM, values: Dict, prior: ObservationalDataLoader, fixed_graph: bool = False):
    log_likelihood = scm.log_likelihood(values, values['y']).item()
    scm_prob = prior.noise_log_prob(scm)
    prior_log_prob = scm_prob
    if not fixed_graph:
        graph = scm.dag
        graph_prob = prior.graph_log_prob(len(graph.nodes), len(graph.edges))
        graph_dropout_prob = prior.graph_dropout_log_prob(len(graph.nodes), len([v for v in graph.nodes if graph.nodes[v].get("hidden", False)]))
        prior_log_prob += graph_prob + graph_dropout_prob
    print(f"Proposal prior: {prior_log_prob}, proposal likelihood: {log_likelihood}")
    return log_likelihood + prior_log_prob


@torch.no_grad()
def perturbate(incumbent_scm: SCM, generator: torch.Generator, perturbation_probs=(1/3, 1/3, 1/3)) -> SCM:
    graph_perturbation_prob, mechanism_perturbation_prob, noise_perturbation_prob = perturbation_probs
    selector = torch.rand((1,), generator=generator)
    if selector < graph_perturbation_prob:
        # Perturb the graph by removing or switching direction of random edge
        new_dag = incumbent_scm.dag.copy()
        nodes = list(new_dag.nodes())
        edges = list(combinations(nodes, 2))
        edge_idx = int(torch.randint(0, len(edges), (1,), generator=generator).item())
        u, v = edges[edge_idx]
        choice = torch.randint(0, 2, (1,), generator=generator).item()
        if new_dag.has_edge(u, v):
            if choice == 0:
                new_dag.remove_edge(u, v)
            else:
                new_dag.remove_edge(u, v)
                new_dag.add_edge(v, u)
        elif new_dag.has_edge(v, u):
            if choice == 0:
                new_dag.remove_edge(v, u)
            else:
                new_dag.remove_edge(v, u)
                new_dag.add_edge(u, v)
        else:
            if choice == 0:
                new_dag.add_edge(u, v)
            else:
                new_dag.add_edge(v, u)
        # Only follow through with change if new_dag is still a valid DAG
        valid_dag = True
        if not nx.is_directed_acyclic_graph(new_dag):
            valid_dag = False
        if len(new_dag.edges()) == 0:
            valid_dag = False
        if new_dag.in_degree('y') == 0 or new_dag.out_degree('y') == 0:
            valid_dag = False
        if valid_dag:
            print("Perturbed graph...")
            new_scm = SCM(new_dag, incumbent_scm.mechanisms, incumbent_scm.noise, generator)
            return new_scm
    elif selector < graph_perturbation_prob + mechanism_perturbation_prob:
        # Switch a random mechanism's activation or perturb its weights
        nodes = list(incumbent_scm.dag.nodes())
        node_idx = int(torch.randint(0, len(nodes), (1,), generator=generator).item())
        v = nodes[node_idx]
        choice = torch.randint(0, 2, (1,), generator=generator).item()
        new_mechanisms = {}
        for v in nodes:
            new_mechanisms[v] = deepcopy(incumbent_scm.mechanisms[v])
        if choice == 0: # switch activation
            new_mechanism = SimpleMechanism(nodes, generator=generator)
            new_mechanisms[v].activation._module[1] = new_mechanism.activation._module[1]
        else: # perturb weight
            parents = list(incumbent_scm.dag.predecessors(v))
            parent_idx = int(torch.randint(0, len(parents), (1,), generator=generator).item())
            w = parents[parent_idx]
            incumbent = new_mechanisms[v].weights[w].item()
            proposal = uniform_proposal(incumbent, -1, 1, 0.1, generator)
            new_mechanisms[v].weights[w].fill_(proposal)
        print("Perturbed mechanisms...")
        new_scm = SCM(incumbent_scm.dag, new_mechanisms, incumbent_scm.noise, generator)
        return new_scm
    elif selector < graph_perturbation_prob + mechanism_perturbation_prob + noise_perturbation_prob:
        # Perturb the noise of a node
        nodes = list(incumbent_scm.dag.nodes())
        new_noise = {v: incumbent_scm.noise[v] for v in nodes}
        node_idx = int(torch.randint(0, len(nodes), (1,), generator=generator).item())
        v = nodes[node_idx]
        incumbent = new_noise[v].distribution.scale
        proposal = uniform_proposal(incumbent, 0, 10, 0.1, generator)
        loc = new_noise[v].distribution.loc
        new_noise[v] = TorchDistributionSampler(dist.Normal(loc=loc, scale=proposal))
        print("Perturbed noise...")
        new_scm = SCM(incumbent_scm.dag, incumbent_scm.mechanisms, new_noise, generator)
        return new_scm
    print("Returning original SCM...")
    return incumbent_scm


def uniform_proposal(incumbent: float, low: float, high: float, max_change: float, generator: torch.Generator) -> float:
    change = torch.rand((1,), generator=generator).item() * 2 * max_change - max_change
    proposal = incumbent + change
    if proposal >= low and proposal < high:
        return proposal
    else:
        return incumbent


def print_mechanisms(scm, filename: str):
    module_dict = scm.mechanisms
    with open(filename, "w") as f:
        for key, module in module_dict.items():
            f.write(f"\n{key}")
            f.write(f"\n{str(module.activation._module[1])}")
            f.write(f"\nNoise std: {scm.noise[key].distribution.scale}")
            weights = [str(w[1].item()) for w in module.weights.items()]
            for w in weights:
                f.write(f"\n{w}")
                
def print_chain(chain: List[Tuple[SCM, float]], output_dir: str):
    for i, (scm, log_prob) in enumerate(chain):
        print(f"Chain member {i}")
        print(f"log probability: {log_prob}")
        print_mechanisms(scm, f"{output_dir}/scm_{i}.py")
        plot_graph(scm.dag, f"{output_dir}/graph_{i}.png")
                

def visualize_chain(chain: List[Tuple[SCM, float]], output_dir: str):
    average_graph = nx.DiGraph()
    average_graph.add_nodes_from(chain[0][0].dag.nodes())
    edge_counts = Counter()
    for scm, _ in chain:
        edge_counts.update((u, v) for u, v in scm.dag.edges())
    num_samples = len(chain)
    for (u, v), count in edge_counts.items():
        probability = count / num_samples
        average_graph.add_edge(u, v, weight=probability)
    plt.figure()
    plot_graph(average_graph, f"{output_dir}/average_graph.png", **DRAWING_STYLE)
    

def mcmc_suite(generator: torch.Generator, output_dir: str, include_mcmc: bool = True, include_pfn: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample the training data D = (X, y)
    graph_samplers = build_samplers(prior_config['graph_config'], "graph")
    scm_samplers = build_samplers(prior_config['scm_config'], "scm")
    graph_params = sample_parameters(graph_samplers, generator)
    scm_params = sample_parameters(scm_samplers, generator)
    graph_builder = GraphBuilder(**graph_params)
    graph = graph_builder.sample(generator)
    scm_builder = SCMBuilder(graph, **scm_params)
    scm = scm_builder.sample(generator)
    sample_shape = (10,)
    scm.sample_noise(sample_shape, generator=generator)
    values = scm.propagate()
    plt.figure()
    plot_graph(graph, f"{output_dir}/true_graph.png", **DRAWING_STYLE)
    print_mechanisms(scm, f"{output_dir}/true_scm.py")
    
    test_sample_shape = (1,)
    scm.sample_noise(test_sample_shape, generator=generator)
    test_sample = scm.propagate()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(x=test_sample['y'].item(), color='black', linestyle='--', linewidth=1, label="True Value")
    
    # adaptively find good range for y
    model_path = 'workdir/ppd'
    model = init_model_from_state_dict_file('pfn', f"{model_path}/latest_checkpoint.pth")
    buckets = torch.load(f"{model_path}/dist.pth")
    bar_dist = FullSupportBarDistribution(buckets)
    reg = Regressor(model, bar_dist, get_default_device())
    y_explore = np.linspace(-15.0, 15.0, 300)
    nodelist = [v for v in test_sample.keys() if v != 'y']
    X_test = torch.stack([test_sample[v] for v in nodelist], dim=-1).cpu().numpy()
    X_train = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
    y_train = values['y'].cpu().numpy()
    reg.fit(X_train, y_train)
    log_probs_explore = reg.log_ppd(X_test, y_explore)
    probs_explore = np.exp(log_probs_explore)
    eps = 1e-2
    mask = probs_explore > eps
    indices = np.where(mask)[0]
    buffer = 1
    start_idx = max(0, indices[0] - buffer)
    end_idx = min(len(y_explore) - 1, indices[-1] + buffer)
    a = y_explore[start_idx]
    b = y_explore[end_idx]
    
    if include_mcmc:
        # Perform MCMC
        prior = ObservationalDataLoader(50, 1, prior_config, seed=seed+1)
        initial_scm = next(iter(prior))['graph_information']['scm']
        chain = mcmc(values, prior, scm, generator)
        
        # Evaluate PPD on test sample
        visualize_chain(chain, output_dir)
        style = {'label': 'ppd', 'color': 'blue', 'linestyle': '-'}
        plot_ppd(ax, test_sample, [scm for scm, _ in chain], style=style, bounds=[a, b])
        
        # Perform MCMC with graph conditioning
        fixed_graph_chain = mcmc(values, prior, scm, generator, fixed_graph=True)
        style = {'label': 'graph_conditioned_ppd', 'color': 'red', 'linestyle': '-'}
        plot_ppd(ax, test_sample, [scm for scm, _ in fixed_graph_chain], style=style, bounds=[a, b])
    
    if include_pfn:
        # Evaluate PPD with PFN
        nodelist = [v for v in values.keys() if v != 'y']
        X_train = torch.stack([values[v] for v in nodelist], dim=-1).cpu().numpy()
        y_train = values['y'].cpu().numpy()
        style = {'label': 'pfn', 'color': 'blue', 'linestyle': '--'}
        plot_ppd_pfn(ax, test_sample, X_train, y_train, reg, style=style, bounds=[a, b])
        
        # Evaluate with PFN incorporating graph knowledge
        model_path = 'workdir/ppd_graph'
        model = init_model_from_state_dict_file('binary', f"{model_path}/latest_checkpoint.pth")
        buckets = torch.load(f"{model_path}/dist.pth")
        bar_dist = FullSupportBarDistribution(buckets)
        reg = Regressor(model, bar_dist, get_default_device())
        nodelist.append('y')
        adjacency_matrix = torch.from_numpy(nx.to_numpy_array(graph, nodelist=nodelist)).to(torch.float32)
        reg = Regressor(model, bar_dist, get_default_device())
        style = {'label': 'graph_pfn', 'color': 'red', 'linestyle': '--'}
        plot_ppd_pfn(ax, test_sample, X_train, y_train, reg, style=style, bounds=[a, b], adjacency_matrix=adjacency_matrix)
        
    ax.set_xlabel("y")
    ax.set_ylabel("p(y)")
    ax.set_title("PPD Comparison")
    ax.legend()
    ax.grid(True)
    fig.savefig(f"{output_dir}/ppds.png", dpi=300)
        

if __name__ == "__main__":
    from configs.ppd_configs import prior_config
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"evaluation/output/{datetime_str}"
    
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    for i in range(20):
        mcmc_suite(generator, f"{output_dir}/run_{i}", include_mcmc=True, include_pfn=True)
        
    
        
        