from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import graph_config, scm_config, preprocessing_config, dataset_config
from probing.plot_correlation import plot_feature_correlation


dataloader = ObservationalDataLoader(1, 1, graph_config, scm_config, preprocessing_config, dataset_config, seed=48)

for data in dataloader:
    X = data['x'][0]
    plot_feature_correlation(X)