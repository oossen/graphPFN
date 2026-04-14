"""
Default configuration for causal discovery.

Adjust the parameters here before running run_discovery.py, or import
`get_default_config()` from your own script.
"""

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from causal_discovery.utils.dot_dict import make_dotdict


def get_default_config():
    """Return the default causal discovery configuration as a DotDict.

    Tune the parameters below to trade off speed vs. thoroughness:

    - n_runs:    Number of random sub-sampling runs per algorithm.
                 More runs → more stable results, but slower.
    - n_samples: Rows sampled per run (capped at the dataset size).
    - n_features: Columns sampled per run (capped at the dataset width).
    - max_time_per_discovery: Wall-clock timeout in seconds per algorithm.
    """
    return make_dotdict(
        {
            # Hard wall-clock timeout (seconds) applied to each discoverer.
            # If a single algorithm exceeds this limit it is cancelled and the
            # ensemble continues with the remaining algorithms.
            "max_time_per_discovery": 60 * 20,  # 20 minutes
            "discoverers": [
                # ------------------------------------------------------------------
                # PC (Peter-Clark) — constraint-based algorithm.
                # Fast and reliable on continuous, low-dimensional data.
                # alpha:      Significance threshold for the conditional-independence
                #             tests. Lower = fewer (but more confident) edges.
                # indep_test: Statistical test used for conditional independence.
                #             "fisherz" works for continuous Gaussian data;
                #             "chisq"/"gsq" are for discrete data.
                {
                    "name": "PCDiscoverer",
                    # Number of sub-sampling runs — more = stabler estimate.
                    "n_runs": 50,
                    # Rows drawn per run (capped at dataset size).
                    "n_samples": 1000,
                    # Columns drawn per run (capped at dataset width).
                    "n_features": 50,
                    # Set to a hyperparameter name (string) to weight runs by
                    # that parameter value; None = uniform weighting.
                    "weight_parameter": None,
                    "config_space": ConfigurationSpace(
                        [
                            UniformFloatHyperparameter(
                                "alpha", lower=0.01, upper=0.1, default_value=0.05
                            ),
                            CategoricalHyperparameter(
                                "indep_test",
                                choices=["fisherz", "chisq", "gsq"],
                                default_value="fisherz",
                            ),
                            Constant(name="show_progress", value=False),
                        ]
                    ),
                },
                # ------------------------------------------------------------------
                # FCI (Fast Causal Inference) — like PC but handles hidden
                # confounders by allowing bidirected edges in the output.
                # Slightly slower than PC.
                {
                    "name": "FCIDiscoverer",
                    "n_runs": 50,
                    "n_samples": 1000,
                    "n_features": 50,
                    "weight_parameter": None,
                    "config_space": ConfigurationSpace(
                        [
                            UniformFloatHyperparameter(
                                "alpha", lower=0.01, upper=0.1, default_value=0.05
                            ),
                            CategoricalHyperparameter(
                                "indep_test",
                                choices=["fisherz", "chisq", "gsq"],
                                default_value="fisherz",
                            ),
                            Constant(name="verbose", value=False),
                            Constant(name="show_progress", value=False),
                        ]
                    ),
                },
                # ------------------------------------------------------------------
                # GES (Greedy Equivalence Search) — score-based algorithm.
                # Does not require a significance threshold; instead it greedily
                # optimises a BIC/BDeu score.  Good complement to PC/FCI.
                # Comment out this block to disable.
                {
                    "name": "GESDiscoverer",
                    "n_runs": 25,
                    "n_samples": 500,
                    "n_features": 20,
                    "weight_parameter": None,
                    # score_func: "local_score_BIC"  — penalises model complexity
                    #             (good default for continuous data).
                    #             "local_score_BDeu" — Bayesian Dirichlet equivalent
                    #             uniform score (better for discrete data).
                    "config_space": ConfigurationSpace(
                        [
                            CategoricalHyperparameter(
                                "score_func",
                                choices=["local_score_BIC", "local_score_BDeu"],
                                default_value="local_score_BIC",
                            ),
                        ]
                    ),
                },
                # ------------------------------------------------------------------
                # ICA-LiNGAM — assumes non-Gaussian noise and uses Independent
                # Component Analysis to find a causal ordering.
                # Best suited when data is clearly non-Gaussian.
                # Comment out this block to disable.
                {
                    "name": "ICALingamDiscoverer",
                    "n_runs": 50,
                    "n_samples": 1000,
                    "n_features": 20,
                    "weight_parameter": None,
                    # max_iter: maximum ICA iterations per run.
                    # Increase if results look unstable.
                    "config_space": ConfigurationSpace(
                        [
                            UniformIntegerHyperparameter(
                                "max_iter", lower=10, upper=1000, default_value=500
                            ),
                        ]
                    ),
                },
            ],
        }
    )


def get_quick_config():
    """Faster config with fewer runs — good for prototyping or testing.

    Uses PC only with a small number of runs so results come back in seconds
    rather than minutes.  Not suitable for production use.
    """
    return make_dotdict(
        {
            # Shorter timeout because we expect runs to finish quickly.
            "max_time_per_discovery": 60 * 5,  # 5 minutes
            "discoverers": [
                # PC only — fastest algorithm, good enough for a sanity check.
                {
                    "name": "PCDiscoverer",
                    # Fewer runs for speed; results will be noisier than default.
                    "n_runs": 10,
                    "n_samples": 500,
                    "n_features": 20,
                    "weight_parameter": None,
                    # Only Fisher-Z in quick mode (fewest degrees of freedom).
                    "config_space": ConfigurationSpace(
                        [
                            UniformFloatHyperparameter(
                                "alpha", lower=0.01, upper=0.1, default_value=0.05
                            ),
                            CategoricalHyperparameter(
                                "indep_test",
                                choices=["fisherz"],
                                default_value="fisherz",
                            ),
                            Constant(name="show_progress", value=False),
                        ]
                    ),
                },
                # To also run FCI in quick mode, uncomment the block below.
                # {
                #     "name": "FCIDiscoverer",
                #     "n_runs": 10,
                #     "n_samples": 500,
                #     "n_features": 20,
                #     "weight_parameter": None,
                #     "config_space": ConfigurationSpace(
                #         [
                #             UniformFloatHyperparameter(
                #                 "alpha", lower=0.01, upper=0.1, default_value=0.05
                #             ),
                #             CategoricalHyperparameter(
                #                 "indep_test",
                #                 choices=["fisherz"],
                #                 default_value="fisherz",
                #             ),
                #             Constant(name="verbose", value=False),
                #             Constant(name="show_progress", value=False),
                #         ]
                #     ),
                # },
            ],
        }
    )
