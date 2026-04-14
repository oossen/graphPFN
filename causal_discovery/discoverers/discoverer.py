from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

from ConfigSpace import ConfigurationSpace
from networkx import Graph
import numpy as np
from numpy import ndarray
from tqdm import tqdm


class Discoverer(ABC):
    def __init__(
        self,
        n_runs: int,
        n_samples,
        n_features,
        weight_parameter: str,
        config_space: ConfigurationSpace,
        rng: int,
    ):
        self.n_runs = n_runs
        self.n_samples = n_samples
        self.n_features = n_features
        self.weight_parameter = weight_parameter
        self.config_space = config_space
        self.rng = rng
        self.discoverer_seed = self.rng.integers(0, 1e9)

    @abstractmethod
    def _process_single_run(self, run_index, data_array, time_limit):
        pass

    def _create_worker_seed(self, discoverer_seed, run_idx):
        return (discoverer_seed * (run_idx + 1)) % (2**32)

    def discover_adjacency_matrices(
        self,
        data_array: ndarray,
        *,
        time_limit: int = 60,
        num_workers: int = 0,
    ):
        adjacency_matrices = []
        weight_parameters = []
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_feature_idx = {}

                reduce_if_n_runs_crash = 5
                for run_idx in range(self.n_runs):
                    worker_seed = self._create_worker_seed(
                        discoverer_seed=self.discoverer_seed,
                        run_idx=run_idx,
                    )
                    sub_data_array, feature_idx = self._random_subsamples(
                        data_array=data_array,
                        worker_seed=worker_seed,
                    )

                    future = executor.submit(
                        self._process_single_run,
                        data_array=sub_data_array,
                        worker_seed=worker_seed,
                        time_limit=time_limit,
                    )
                    future_to_feature_idx[future] = feature_idx

                crashed_runs_counter = 0
                for future in tqdm(as_completed(future_to_feature_idx), total=self.n_runs, desc=type(self).__name__, unit="run", leave=False):
                    try:
                        result = future.result()
                        if result is not None:
                            adjacency_matrices.append(
                                self._expand_adjacency_matrix(
                                    max_features=data_array.shape[1],
                                    collapsed_adjacency_matrix=result[0],
                                    feature_idx=future_to_feature_idx[future],
                                )
                            )
                            weight_parameters.append(result[1])
                        else:
                            crashed_runs_counter += 1

                            if reduce_if_n_runs_crash == crashed_runs_counter:
                                crashed_runs_counter = 0
                                old_n_features = self.n_features
                                self.n_features = max(int(self.n_features * 0.75), 5)
                                time_limit = max(int(2.5 * 60), int(time_limit * 0.5))

                                print(
                                    f"Run failed — reducing from {old_n_features} to {self.n_features} features."
                                )

                    except Exception as e:
                        print(f"Worker failed with exception: {e}")

        else:
            pbar = tqdm(range(self.n_runs), desc=type(self).__name__, unit="run", leave=False)
            for run_idx in pbar:
                sub_data_array, feature_idx = self._random_subsamples(
                    data_array=data_array,
                    worker_seed=run_idx,
                )
                worker_seed = self._create_worker_seed(
                    discoverer_seed=self.discoverer_seed,
                    run_idx=run_idx,
                )
                res = self._process_single_run(
                    data_array=sub_data_array,
                    worker_seed=worker_seed,
                    time_limit=time_limit,
                )
                if res is None:
                    continue
                adjacency_matrices.append(
                    self._expand_adjacency_matrix(
                        max_features=data_array.shape[1],
                        collapsed_adjacency_matrix=res[0],
                        feature_idx=feature_idx,
                    )
                )
                if self.weight_parameter:
                    weight_parameters.append(res[1])

        valid = True
        if len(adjacency_matrices) == 0:
            probabilistic_adjacency = None
            stacked_adjacencies = None
            weight_parameters = None
            weights = None
            valid = False
        else:
            if not self.weight_parameter:
                weight_parameters = [1.0] * len(adjacency_matrices)
            probabilistic_adjacency, weights = self.weight_linearly(
                weight_parameters, adjacency_matrices
            )
            stacked_adjacencies = np.stack(adjacency_matrices, axis=0)

        return {
            "probabilistic_adjacencies": probabilistic_adjacency,
            "stacked_adjacencies": stacked_adjacencies,
            "weight_parameters": weight_parameters,
            "weights": weights,
            "valid": valid,
        }

    def _expand_adjacency_matrix(
        self,
        max_features,
        collapsed_adjacency_matrix,
        feature_idx,
    ):
        full_adjacency_matrix = np.zeros((max_features, max_features), dtype=int)

        for i in range(len(feature_idx)):
            for j in range(len(feature_idx)):
                full_adjacency_matrix[feature_idx[i], feature_idx[j]] = (
                    collapsed_adjacency_matrix[i, j]
                )

        return full_adjacency_matrix

    def graph2adjacency(self, graph: Graph):
        num_nodes = len(graph.nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge = graph.get_edge(graph.nodes[i], graph.nodes[j])
                if edge is not None:
                    adjacency_matrix[i, j] = 1

        return adjacency_matrix

    def _random_subsamples(self, data_array: ndarray, worker_seed) -> ndarray:
        n_samples = min(self.n_samples, data_array.shape[0])
        n_features = min(self.n_features, data_array.shape[1])
        rng = np.random.default_rng(worker_seed)
        row_indices = rng.choice(data_array.shape[0], n_samples, replace=False)
        col_indices = sorted(rng.choice(data_array.shape[1], n_features, replace=False))
        subsample = data_array[row_indices][:, col_indices]
        return subsample, col_indices

    def weight_linearly(
        self,
        weight_parameters: list,
        adjacency_matrices: list,
    ):
        if len(weight_parameters) != len(adjacency_matrices):
            raise ValueError(
                "weight_parameters and adjacency_matrices must have the same length"
            )

        weights = np.array([1 / para for para in weight_parameters])
        weights /= np.sum(weights)

        return sum(w * A for w, A in zip(weights, adjacency_matrices)), weights
