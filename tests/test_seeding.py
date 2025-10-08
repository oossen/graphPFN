from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import graph_config, scm_config, preprocessing_config, dataset_config
from probing.compare_dataloaders import compare_dataloaders


loader_1 = ObservationalDataLoader(3, 10, graph_config, scm_config, preprocessing_config, dataset_config, seed=42)
loader_2 = ObservationalDataLoader(3, 10, graph_config, scm_config, preprocessing_config, dataset_config, seed=42)

compare_dataloaders(loader_1, loader_2)