from copy import deepcopy
from pathlib import Path
import signal
import time

from causal_discovery.discoverers.fci_discoverer import FCIDiscoverer
from causal_discovery.discoverers.ges_discoverer import GESDiscoverer
from causal_discovery.discoverers.ica_lingam_discoverer import ICALingamDiscoverer
from causal_discovery.discoverers.pc_discoverer import PCDiscoverer
import numpy as np
from numpy.random import default_rng
import pandas as pd
from tqdm import tqdm
from causal_discovery.utils.dot_dict import DotDict, make_dotdict as Dictionary
from causal_discovery.utils.io_utils import load_array, save_array
from causal_discovery.utils.plot_utils import plot_adjacency_heatmap, plot_graph_from_adjacency_matrix


class CausalExplorer:
    def __init__(self, config: DotDict, results_path, rng):
        self.config = config
        self.results_path = results_path
        self.rng = rng

        self.causal_discoverers = self._get_discoverers(self.config.discoverers)
        self.results = []

        # Register signal handler for graceful termination
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def _get_discoverers(self, discoverers_dict: str):
        discoverer_classes = {
            "pcdiscoverer": PCDiscoverer,
            "fcidiscoverer": FCIDiscoverer,
            "gesdiscoverer": GESDiscoverer,
            "icalingamdiscoverer": ICALingamDiscoverer,
        }

        discoverers_list = []
        for discoverers_config in discoverers_dict:
            config = Dictionary(discoverers_config)
            discoverer_name = config.name.lower()

            DiscovererClass = discoverer_classes.get(discoverer_name)
            if DiscovererClass:
                sub_rng = default_rng(self.rng.integers(0, 1e9))
                discoverers_list.append(
                    DiscovererClass(
                        n_runs=config.n_runs,
                        n_samples=config.n_samples,
                        n_features=config.n_features,
                        weight_parameter=config.weight_parameter,
                        config_space=config.config_space,
                        rng=sub_rng,
                    )
                )

        return discoverers_list

    def signal_handler(self, signum, frame):
        """Handle signals from the operating system or job scheduler."""
        if not self.results:
            return

        self._aggregate_and_save_results()

    def _aggregate_and_save_results(self):
        """Save processed results to disk."""
        if not self.results:
            return None

        results_dir = Path(self.results_path).resolve()
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        self.end_time = time.time()
        causal_results = self._aggregate_across_discoverers(self.results)

        save_config = deepcopy(self.config)
        save_config["start_time"] = self.start_time
        save_config["end_time"] = self.end_time
        save_config["time_taken"] = self.end_time - self.start_time
        save_config["seed"] = self.rng

        config_path = Path(results_dir) / "config.json"
        with open(config_path, "w") as f:
            f.write(str(save_config))

        for key, value in causal_results.items():
            save_path = Path(results_dir) / f"{key}.npy"
            save_array(path=save_path, array=value)

        viz_path = Path(results_dir) / "probabilistic_adjacency.png"
        plot_adjacency_heatmap(
            adjacency_matrix=causal_results.probabilistic_adjacency,
            title="Probabilistic Adjacency Matrix",
            path=viz_path,
        )

        graph_path = Path(results_dir) / "probabilistic_adjacency_graph.png"
        plot_graph_from_adjacency_matrix(
            adjacency_matrix=causal_results.probabilistic_adjacency,
            title="All likely edges",
            path=graph_path,
        )
        return causal_results

    def _aggregate_across_discoverers(self, results):
        """Processes raw results into a structured DotDict."""
        if not results:
            raise ValueError("No results to process")

        results = [r if isinstance(r, DotDict) else Dictionary(r) for r in results]

        probabilistic_adjacencies = [
            result.probabilistic_adjacencies
            for result in results
            if hasattr(result, "probabilistic_adjacencies")
        ]

        if not probabilistic_adjacencies:
            raise ValueError("No valid probabilistic adjacencies found in results")

        probabilistic_adjacency = np.sum(probabilistic_adjacencies, axis=0) / len(
            probabilistic_adjacencies
        )

        stacked_adjacencies = []
        weight_parameters = []
        weights = []

        for result in results:
            if hasattr(result, "stacked_adjacencies"):
                stacked_adjacencies.append(result.stacked_adjacencies)
            if hasattr(result, "weight_parameters"):
                weight_parameters.extend(result.weight_parameters)
            if hasattr(result, "weights"):
                weights.extend(result.weights)

        if stacked_adjacencies:
            stacked_adjacencies = np.concatenate(stacked_adjacencies, axis=0)
        else:
            stacked_adjacencies = np.array([])

        weight_parameters = (
            np.array(weight_parameters).flatten() if weight_parameters else np.array([])
        )
        weights = np.array(weights).flatten() if weights else np.array([])

        np.fill_diagonal(probabilistic_adjacency, 0)

        return Dictionary(
            {
                "probabilistic_adjacency": probabilistic_adjacency,
                "stacked_adjacencies": stacked_adjacencies,
                "weight_parameters": weight_parameters,
                "weights": weights,
            }
        )

    def discover_causal_structures(self, data_df: pd.DataFrame, num_workers=0) -> dict:
        """Runs causal discovery on the given DataFrame."""
        data_array = data_df.values

        self.results = []
        self.start_time = time.time()

        for discoverer in tqdm(self.causal_discoverers, desc="Ensemble", unit="algo"):
            result = discoverer.discover_adjacency_matrices(
                data_array=data_array,
                time_limit=min(5 * 60, self.config.max_time_per_discovery),
                num_workers=num_workers,
            )
            if result and result.get("valid", False):
                self.results.append(result)

        end_time = time.time() - self.start_time
        print(
            f"Ensemble time: {end_time:.2f} seconds [{end_time / 60:.1f} min], num_workers: {num_workers}"
        )
        return self._aggregate_and_save_results()

    @staticmethod
    def load_causal_results(
        results_path,
        data_df=None,
        causal_discovery_config=None,
        num_workers=0,
        rng=None,
    ):
        """Load cached results or run discovery if no results exist yet.

        Args:
            results_path (Path): Directory to load/save results.
            data_df (pd.DataFrame): Input data (required if no cached results exist).
            causal_discovery_config (DotDict): Configuration (required if no cached results).
            num_workers (int): Number of parallel workers (0 = sequential).
            rng: NumPy random generator instance.

        Returns:
            DotDict with keys: probabilistic_adjacency, stacked_adjacencies,
                               weight_parameters, weights.
        """
        Path(results_path).mkdir(parents=True, exist_ok=True)

        npy_files = [f for f in results_path.iterdir() if f.suffix == ".npy"]

        if not any(npy_files):
            explorer = CausalExplorer(
                config=causal_discovery_config,
                results_path=results_path,
                rng=rng,
            )
            causal_results = explorer.discover_causal_structures(
                data_df=data_df,
                num_workers=num_workers,
            )
        else:
            causal_results = DotDict({f.stem: load_array(f) for f in npy_files})

        return causal_results
