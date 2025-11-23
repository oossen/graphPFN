from typing import Optional, Tuple

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


    def sample_ER_DAG(self, generator: Optional[torch.Generator]) -> Tuple[nx.DiGraph, np.ndarray]:
        """
        Create a random DAG.

        Parameters
        ----------
        generator : torch.Generator
            Used to make sampling of graphs deterministic.

        Returns
        -------
        G : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        adj : torch.Tensor
            The probabilistic adjacency matrix underlying G.
        """
        # Get numpy generator from torch generator
        np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
        self.rng = np.random.default_rng(np_seed)
        
        n = int(self.num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 < self.edge_prob < 1.0):
            raise ValueError("p must be in (0, 1).")

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        # Random topological order
        perm = self.rng.permutation(n)
        
        # probabilistic matrix
        beta = 0.5
        alpha = (self.edge_prob * beta) / (1 - self.edge_prob) # mean of distribution is at self.edge_prob
        adj = self.rng.beta(a=alpha, b=beta, size=(n, n))
        adj = np.triu(adj, k=1)
        adj[perm[:, None], perm] = adj.copy()
        mask = self.rng.random((n, n)) < adj

        # Extract and add edges
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size:
            src = i_idx
            dst = j_idx
            G.add_edges_from(zip(src.tolist(), dst.tolist()))
            
        # resample if there are no edges
        if len(G.edges) == 0:
            return self.sample_ER_DAG(generator)

        return G, adj