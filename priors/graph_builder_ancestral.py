from copy import deepcopy
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

    def __init__(self, num_nodes: int, edge_prob: float, dropout_prob: float) -> None:
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        edge_prob : float
            Probability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].
        dropout_prob : float
            Probability of making a given node hidden.
        """
        self.num_nodes = num_nodes
        # Set a minimum probability to avoid very sparse small graphs
        # 2 -> 87%, 3 -> 54%, 5 -> 28%, 10 -> 13%, 20 -> 5% 30 -> 3%
        edge_prob_min = 2 / (num_nodes ** 1.2)
        self.edge_prob = max(edge_prob_min, edge_prob) 
        self.dropout_prob = dropout_prob


    def sample_graph(self, generator: Optional[torch.Generator]) -> Tuple[nx.DiGraph, nx.DiGraph, nx.Graph]:
        """
        Create a random DAG. First samples a probabilistic graph, from which a binary graph is then sampled.
        Finally, a number of nodes are hidden and an undirected graph recording the confounding is created.

        Parameters
        ----------
        generator : torch.Generator
            Used to make sampling of graphs deterministic.

        Returns
        -------
        graph : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        new_graph : nx.DiGraph
            The graph obtained from `graph` by contracting hiddn nodes.
        confounding_graph : nx.Graph
            An undirected graph recording the confounding structure induced by hidden nodes.
            
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
        beta = 0.5
        alpha = (self.edge_prob * beta) / (1 - self.edge_prob) # mean of distribution is at self.edge_prob
        adj = self.rng.beta(a=alpha, b=beta, size=(n, n))
        adj = np.triu(adj, k=1)
        adj[perm[:, None], perm] = adj.copy()
        mask = self.rng.random((n, n)) < adj

        graph = nx.from_numpy_array(mask, create_using=nx.DiGraph)
        probabilistic_graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
            
        # resample if there are no edges
        if len(graph.edges) == 0:
            return self.sample_graph(generator)
        
        # Hide some nodes
        attribute_dict = {
            v: torch.rand(1, generator=generator) < self.dropout_prob 
            for v in graph.nodes
        }
        nx.set_node_attributes(graph, attribute_dict, name="hidden")
        
        # select target and rename
        visible_nodes = [v for v in graph.nodes if not graph.nodes[v]["hidden"]]
        # resample if less than 2 visible nodes
        if len(visible_nodes) < 2:
            return self.sample_graph(generator)
        hidden_nodes = [v for v in graph.nodes if graph.nodes[v]["hidden"]]
        target_node_idx = int(torch.randint(0, len(visible_nodes), (1,), generator=generator))
        target_node = visible_nodes[target_node_idx]
        renaming = {}
        for v in hidden_nodes:
            renaming[v] = f"u{str(v)}"
        for v in visible_nodes:
            if v != target_node:    
                renaming[v] = f"x{str(v)}"
        renaming[target_node] = "y"
        graph = nx.relabel_nodes(graph, renaming)
        probabilistic_graph = nx.relabel_nodes(probabilistic_graph, renaming)
        
        # construct new graph and confounding graph
        # define visible and hidden nodes again because their names have changed
        visible_nodes = [v for v in graph.nodes if not graph.nodes[v]["hidden"]]
        hidden_nodes = [v for v in graph.nodes if graph.nodes[v]["hidden"]]
        new_graph = deepcopy(probabilistic_graph)
        confounding_graph = nx.Graph()
        confounding_graph.add_nodes_from(visible_nodes)
        for v in hidden_nodes:
            succs = [u for u in probabilistic_graph.successors(v)]
            preds = [u for u in probabilistic_graph.predecessors(v)]
            for u2 in succs:
                for u1 in preds:
                    if u1 != u2:
                        old_weight = probabilistic_graph.edges[u1, u2]['weight']
                        new_weight = probabilistic_graph.edges[u1, v]['weight'] * probabilistic_graph.edges[v, u2]['weight']
                        weight = 1 - (1 - old_weight) * (1 - new_weight)
                        new_graph.add_edge(u1, u2, contracted=True, weight=weight)
                for u1 in succs:
                    if u1 != u2:
                        old_weight = confounding_graph.edges[u1, u2]['weight'] if confounding_graph.has_edge(u1, u2) else 0.0
                        new_weight = probabilistic_graph.edges[v, u1]['weight'] * probabilistic_graph.edges[v, u2]['weight']
                        weight = 1 - (1 - old_weight) * (1 - new_weight)
                        confounding_graph.add_edge(u1, u2, weight=weight)
        new_graph.remove_nodes_from(hidden_nodes)
        confounding_graph.remove_nodes_from(hidden_nodes)

        return graph, new_graph, confounding_graph