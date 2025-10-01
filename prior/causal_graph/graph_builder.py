import random
from typing import Optional

import numpy as np
import networkx as nx


class GraphBuilder:
    """
    Utility class for generating random DAGs (Directed Acyclic Graphs).
    Acyclicity is ensured by sampling edges only from earlier to later nodes in
    a random topological order (random permutation).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        seed : int, optional
            Seed used for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

    def sample_ER_DAG(
        self,
        num_nodes: int,
        edge_prob: float,
    ) -> nx.DiGraph:
        """
        Create a random DAG.

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        p : float
            Pprobability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].

        Returns
        -------
        G : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        """
        n = int(num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 <= edge_prob <= 1.0):
            raise ValueError("p must be in [0, 1].")

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        if n <= 1 or edge_prob == 0.0:
            return G

        # Random topological order
        perm = self.rng.permutation(n)

        # Strictly upper-triangular Bernoulli mask (acyclic by construction)
        mask = np.triu(self.rng.random((n, n)) < edge_prob, k=1)

        # Extract and add edges
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size:
            src = perm[i_idx]
            dst = perm[j_idx]
            G.add_edges_from(zip(src.tolist(), dst.tolist()))

        return G

    