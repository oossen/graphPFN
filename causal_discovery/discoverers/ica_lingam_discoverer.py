from causal_discovery.discoverers.discoverer import Discoverer
from causallearn.search.FCMBased import lingam
from ConfigSpace import ConfigurationSpace
import numpy as np
from causal_discovery.utils.time_limit import TimeLimit


class ICALingamDiscoverer(Discoverer):
    def __init__(
        self,
        n_runs: int,
        n_samples,
        n_features,
        weight_parameter,
        config_space: ConfigurationSpace,
        rng: int,
    ):
        super().__init__(
            n_runs=n_runs,
            n_samples=n_samples,
            n_features=n_features,
            weight_parameter=weight_parameter,
            config_space=config_space,
            rng=rng,
        )

    def _process_single_run(self, data_array, worker_seed, time_limit):
        self.config_space.random = np.random.RandomState(worker_seed)
        sampled_config = dict(self.config_space.sample_configuration())
        try:
            with TimeLimit(seconds=time_limit):
                model = lingam.ICALiNGAM(
                    random_state=worker_seed,
                    max_iter=sampled_config["max_iter"],
                )
                model.fit(X=data_array)
        except Exception:
            return None
        weight_param = (
            sampled_config[self.weight_parameter]
            if self.weight_parameter
            else None
        )
        adjacency_matrix = (model.adjacency_matrix_ != 0).astype(int)
        return (adjacency_matrix, weight_param)
