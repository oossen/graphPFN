import pygad
import numpy as np
import torch
from typing import List, Dict, Callable

from graphpfn.interface import Regressor


def build_matrix(vector, n):
    """Transforms a 1D gene array into a constrained nxn matrix."""
    matrix = np.zeros((n, n))
    gene_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Extract two genes for the pair (i,j) and (j,i), where i < j
            g1 = vector[gene_idx]
            g2 = vector[gene_idx + 1]
            gene_idx += 2
            # Constraint: a_ij + a_ji <= 1
            total = g1 + g2
            if total > 1.0:
                g1 /= total
                g2 /= total
            matrix[i, j] = g1
            matrix[j, i] = g2
    return matrix


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    total_gens = ga_instance.num_generations
    print(f"Finished Generation {gen}/{total_gens}. "
          f"Best Fitness so far: {ga_instance.best_solution()[1]}")


def run_evolution(tables: List[Dict], 
                  model: Regressor, 
                  metric: Callable, 
                  num_generations: int = 100,
                  num_parents_mating: int = 10,
                  sol_per_pop: int = 20,
                  seed: int = 42):
    """
    Find the optimal probabilistic adjacency matrix for the given model and metric using a genetic algorithm.
    
    Parameters
    ----------
    tables : List[Dict]
        A list of tables, where each table is a dictionary with 'X_train', 'y_train', 'X_test', 'y_test' keys.
        All tables should have the same number of train and test samples for comparability
        and all tables *must* have the same number of features.
    model : Regressor
        The model to evaluate (must be able to handle probabilistic adjacency matrices).
    metric : Callable
        A function that takes (preds, y_test) and returns a scalar score (higher is better!).
    num_generations : int
        Number of generations for the genetic algorithm.
    num_parents_mating : int
        Number of parents to select for mating in each generation.
    sol_per_pop : int
        Population size.
    seed : int
        Random seed for reproducibility.
    """
    n_cols = tables[0]['X_train'].shape[1] + 1
    n_genes = n_cols * (n_cols - 1)  # = n_cols * n_cols minus the diagonal
    
    def fitness_func(ga_instance, solution, solution_idx):
        prob_adj = build_matrix(solution, n_cols)
        prob_adj_tensor = torch.tensor(prob_adj, dtype=torch.float32)

        scores = []
        for table in tables:
            X_train, y_train = table['X_train'], table['y_train']
            X_test, y_test = table['X_test'], table['y_test']
            model.fit(X_train, y_train)
            preds = model.predict(X_test, prob_adj=prob_adj_tensor)
            score = metric(preds, y_test)
            scores.append(score)
        return np.mean(scores)

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=n_genes,
        gene_space={'low': 0, 'high': 1},
        on_generation=on_generation,
        random_seed=seed,
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_matrix = build_matrix(solution, n_cols)
    return best_matrix