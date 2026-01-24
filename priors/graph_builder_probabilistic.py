from typing import Optional
import numpy as np
import networkx as nx
import torch


class GraphBuilder:
    """
    Utility class for generating random DAGs (Directed Acyclic Graphs).
    Acyclicity is ensured by sampling edges only from earlier to later nodes in
    a random topological order (random permutation).
    """

    def __init__(self, num_nodes: int, edge_prob: float) -> None:
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        edge_prob : float
            Probability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].
        """
        self.num_nodes = num_nodes
        # Set a minimum probability to avoid very sparse small graphs
        # 2 -> 87%, 3 -> 54%, 5 -> 28%, 10 -> 13%, 20 -> 5% 30 -> 3%
        edge_prob_min = 2 / (num_nodes ** 1.2)
        self.edge_prob = max(edge_prob_min, edge_prob)
         
    def sample_edge_prob(self, shape, generator: Optional[torch.Generator]) -> np.ndarray:
        """Sample edge weights from a beta distribution."""
        # Get numpy generator from torch generator
        np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
        self.rng = np.random.default_rng(np_seed)
        
        beta = 0.5
        alpha = (self.edge_prob * beta) / (1 - self.edge_prob) # mean of distribution is at self.edge_prob
        edge_prob = self.rng.beta(a=alpha, b=beta, size=shape)
        return edge_prob
    
    def rename(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Rename nodes to x0, x1, ..., y."""
        nodes = list(graph.nodes)
        target_node = nodes[-1]
        renaming = {target_node: 'y'}
        for i, node in enumerate(nodes[:-1]):
            renaming[node] = f'x{i}'
        graph = nx.relabel_nodes(graph, renaming)
        return graph

    def sample(self, generator: Optional[torch.Generator]) -> nx.DiGraph:
        """
        Create a random DAG.

        Parameters
        ----------
        generator : torch.Generator
            Used to make sampling of graphs deterministic.

        Returns
        -------
        G : nx.DiGraph
            The generated DAG with float edge weights.
            
        """
        # Get numpy generator from torch generator
        np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
        self.rng = np.random.default_rng(np_seed)
        
        n = int(self.num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 < self.edge_prob < 1.0):
            raise ValueError("p must be in (0, 1).")

        # Random topological order
        perm = self.rng.permutation(n)
        
        # probabilistic matrix
        adj = self.sample_edge_prob((n, n), generator)
        adj = np.triu(adj, k=1)
        adj[perm[:, None], perm] = adj.copy()

        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        graph = self.rename(graph)

        return graph
    
    
    def perturbate_graph(self, graph: nx.DiGraph, generator: Optional[torch.Generator], resample_prob: float = 0.5) -> nx.DiGraph:
        """
        Return a copy of `graph` with some edge weights resampled.
        """
        new_graph = graph.copy()
        old_adj = nx.to_numpy_array(graph)
        resample_adj = self.sample_edge_prob(old_adj.shape, generator)
        resample_mask = (torch.rand(old_adj.shape, generator=generator) < resample_prob) & torch.from_numpy(old_adj > 0)
        new_adj = np.where(resample_mask.numpy(), resample_adj, old_adj)
        new_graph = nx.from_numpy_array(new_adj, create_using=nx.DiGraph)
        
        new_graph = self.rename(new_graph)
        
        return new_graph