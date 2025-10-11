from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import prior_config, preprocessing_config
from probing.plot_correlation import plot_feature_correlation


dataloader = ObservationalDataLoader(1, 1, prior_config, preprocessing_config, seed=99)

for data in dataloader:
    X = data['x'][0]
    plot_feature_correlation(X)