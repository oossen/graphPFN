from collections import Counter
import math
import os
from typing import Dict, List, Mapping, Tuple
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


@torch.no_grad()
def mcmc(values: Dict,
         prior: ObservationalDataLoader, 
         initial_scm: SCM, 
         generator: torch.Generator, 
         steps: int = 1000, 
         burn_in: int = 0, 
         step_size: int = 2,
         jump_prob: float = 0.1) -> List[Tuple[SCM, float]]:
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
    burn_in : int
        The number of discarded samples at the beginning of the chain.
    step_size : int
        Only every `step_size`-th sample is returned at the end.
    jump_prob : float
        The probability to jump to a random new incumbent instead of perturbating the current one.
    
    Returns
    -------
    chain : List[(SCM, float)]
        The list of accepted samples the algorithm has produced, paired with their posterior log probability.
    """
    incumbent = initial_scm
    incumbent_log_prob = posterior_log_prob(incumbent, values, prior)
    chain = [(incumbent, incumbent_log_prob)]
    for _ in range(steps):
        if torch.rand((1,), generator=generator) < jump_prob:
            print("Jumping to random new incumbent...")
            proposal = next(iter(prior))['graph_information']['scm']
        else:
            proposal = perturbate(incumbent, generator)
        proposal_log_prob = posterior_log_prob(proposal, values, prior)
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
    return chain[burn_in::step_size]


def ppd(values: Dict, samples: List[SCM]) -> float:
    """
    Approximate the PPD of y given the features in `values`
    using Monte Carlo integration with the provided `samples`.
    """
    likelihoods = []
    for scm in samples:
        ll = scm.log_likelihood(values, 'y')
        likelihoods.append(math.exp(ll))
    return sum(likelihoods) / len(likelihoods)


def plot_ppd(values: Dict, samples: List[SCM], filename: str, steps: int = 30):    
    y = np.linspace(-3, 3, steps)
    probs = []
    for yi in y:
        values_y = values | {'y': torch.tensor([yi], dtype=torch.float32)}
        probs.append(ppd(values_y, samples))
    plt.plot(y, probs, label="p(y)")
    plt.axvline(x=values['y'].item(), color='red', linestyle='--', linewidth=1)  # the true y-value
    plt.xlabel("y")
    plt.ylabel("p(y)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename}/ppd.png", dpi=300)
    plt.close()
    

def plot_ppd_pfn(values: Dict, X_train, y_train, model, filename: str, steps: int = 30):
    y = np.linspace(-3, 3, steps)
    model.fit(X_train, y_train)
    X_test = torch.stack([values[v] for v in ['x0', 'x1', 'x2', 'x3']], dim=2).cpu().numpy()
    log_probs = model.log_ppd(X_test, y)
    probs = np.exp(log_probs)
    plt.plot(y, probs, label="p(y)")
    plt.axvline(x=values['y'].item(), color='red', linestyle='--', linewidth=1)  # the true y-value
    plt.xlabel("y")
    plt.ylabel("p(y)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename}/ppd_pfn.png", dpi=300)
    plt.close()
    

@torch.no_grad()    
def posterior_log_prob(scm: SCM, values: Dict, prior: ObservationalDataLoader):
    log_likelihood = scm.log_likelihood(values, 'y')
    graph = scm.dag
    graph_prob = prior.graph_log_prob(len(graph.nodes), len(graph.edges))
    graph_dropout_prob = prior.graph_dropout_log_prob(len(graph.nodes), len([v for v in graph.nodes if graph.nodes[v].get("hidden", False)]))
    scm_prob = prior.noise_log_prob(scm)
    prior_log_prob = scm_prob + graph_prob + graph_dropout_prob
    print(f"Proposal prior: {prior_log_prob}, proposal likelihood: {log_likelihood}")
    return log_likelihood + prior_log_prob


@torch.no_grad()
def perturbate(incumbent_scm: SCM, generator: torch.Generator) -> SCM:
    graph_perturbation_prob, mechanism_perturbation_prob, noise_perturbation_prob = 1/3, 1/3, 1/3
    selector = torch.rand((1,), generator=generator)
    if selector < graph_perturbation_prob:
        # Perturb the graph
        switch_prob = 0.1
        new_dag = incumbent_scm.dag.copy()
        nodes = list(new_dag.nodes())
        for u, v in combinations(nodes, 2):
            if torch.rand((1,), generator=generator) < switch_prob:
                if new_dag.has_edge(u, v):
                    new_dag.remove_edge(u, v)
                if new_dag.has_edge(v, u):
                    new_dag.remove_edge(v, u)
                choice = torch.randint(0, 3, (1,), generator=generator).item()
                if choice == 0:
                    pass
                if choice == 1:
                    new_dag.add_edge(u, v)
                elif choice == 2:
                    new_dag.add_edge(v, u)
        # Only follow through with change if new_dag is still acyclic
        if nx.is_directed_acyclic_graph(new_dag):
            print("Perturbed graph...")
            new_scm = SCM(new_dag, incumbent_scm.mechanisms, incumbent_scm.noise, generator)
            return new_scm
    elif selector < graph_perturbation_prob + mechanism_perturbation_prob:
        # Perturb the mechanisms
        activation_switch_prob = 0.1
        max_weight_change = 0.1
        nodes = list(incumbent_scm.dag.nodes())
        incumbent_mechanisms: Mapping = incumbent_scm.mechanisms
        new_mechanisms = {}
        for v in nodes:
            new_mechanisms[v] = SimpleMechanism(nodes, generator=generator)
            if torch.rand((1,), generator=generator) > activation_switch_prob:
                new_mechanisms[v].activation._module[1] = incumbent_mechanisms[v].activation._module[1]
            for w in new_mechanisms[v].weights:
                incumbent = incumbent_mechanisms[v].weights[w].item()
                proposal = uniform_proposal(incumbent, -1, 1, max_weight_change, generator)
                new_mechanisms[v].weights[w].fill_(proposal)
        print("Perturbed mechanisms...")
        new_scm = SCM(incumbent_scm.dag, new_mechanisms, incumbent_scm.noise, generator)
        return new_scm
    elif selector < graph_perturbation_prob + mechanism_perturbation_prob + noise_perturbation_prob:
        # Perturb the noise
        max_std_change = 0.1
        nodes = list(incumbent_scm.dag.nodes())
        incumbent_noise: Mapping = incumbent_scm.noise
        new_noise = {}
        for v in nodes:
            incumbent = incumbent_noise[v].distribution.scale
            proposal = uniform_proposal(incumbent, 0, 10, max_std_change, generator)
            loc = incumbent_noise[v].distribution.loc
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
    plot_graph(average_graph, f"{output_dir}/average_graph.png")
    

def mcmc_suite(generator: torch.Generator, output_dir: str):
    graph_samplers = build_samplers(prior_config['graph_config'], "graph")
    scm_samplers = build_samplers(prior_config['scm_config'], "scm")
    graph_params = sample_parameters(graph_samplers, generator)
    scm_params = sample_parameters(scm_samplers, generator)
    
    graph_builder = GraphBuilder(**graph_params)
    graph = graph_builder.sample(generator)
    scm_builder = SCMBuilder(graph, **scm_params)
    scm = scm_builder.sample(generator)
    sample_shape = (5,)
    scm.sample_noise(sample_shape, generator=generator)
    values = scm.propagate(sample_shape)
    plot_graph(graph, f"{output_dir}/true_graph.png")
    print_mechanisms(scm, f"{output_dir}/true_scm.py")
    
    prior = ObservationalDataLoader(50, 1, prior_config, seed=seed+1)
    initial_scm = next(iter(prior))['graph_information']['scm']
    chain = mcmc(values, prior, scm, generator)
    
    test_sample_shape = (1,)
    scm.sample_noise(test_sample_shape, generator=generator)
    test_sample = scm.propagate(test_sample_shape)
    visualize_chain(chain, output_dir)
    plot_ppd(test_sample, [scm for scm, _ in chain], output_dir)
    
    model_path = 'workdir/ppd'
    model = init_model_from_state_dict_file('pfn', f"{model_path}/latest_checkpoint.pth")
    buckets = torch.load(f"{model_path}/dist.pth")
    bar_dist = FullSupportBarDistribution(buckets)
    reg = Regressor(model, bar_dist, get_default_device())
    X_train = torch.stack([values[v] for v in ['x0', 'x1', 'x2', 'x3']], dim=2).cpu().numpy()
    y_train = values['y'].cpu().numpy()
    plot_ppd_pfn(test_sample, X_train, y_train, reg, output_dir)
    
        
if __name__ == "__main__":
    from configs.ppd_configs import prior_config
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"evaluation/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    seed = 43
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    for i in range(5):
        mcmc_suite(generator, f"{output_dir}/run_{i}")
        
    
        
        