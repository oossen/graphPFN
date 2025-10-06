from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import default_graph_config, default_dataset_config, default_preprocessing_config, default_scm_config
from probing.plot_correlation import plot_feature_correlation


dataloader = ObservationalDataLoader(1, 1, default_graph_config, default_scm_config, default_preprocessing_config, default_dataset_config, seed=42)

for data in dataloader:
    X = data['x'][0]
    plot_feature_correlation(X)