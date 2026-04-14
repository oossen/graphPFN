from causal_discovery.discoverers.discoverer import Discoverer
from causallearn.search.ConstraintBased.PC import pc
from ConfigSpace import ConfigurationSpace
import numpy as np
from causal_discovery.utils.time_limit import TimeLimit


class PCDiscoverer(Discoverer):
    def __init__(
        self,
        n_runs: int,
        n_samples,
        n_features,
        weight_parameter,
        config_space: ConfigurationSpace,
        rng,
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
        sampled_config = self.config_space.sample_configuration()

        try:
            with TimeLimit(seconds=time_limit):
                result = pc(data=data_array, **sampled_config)
        except:  # noqa: E722
            return None
        weight_param = (
            sampled_config.get(self.weight_parameter, 0.0)
            if self.weight_parameter
            else None
        )
        adjacency_matrix = self.graph2adjacency(result.G)
        return (adjacency_matrix, weight_param)
