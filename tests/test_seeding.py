from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import default_graph_config, default_dataset_config, default_preprocessing_config, default_scm_config
from probing.compare_dataloaders import compare_dataloaders


loader_1 = ObservationalDataLoader(3, 10, default_graph_config, default_scm_config, default_preprocessing_config, default_dataset_config, seed=42)
loader_2 = ObservationalDataLoader(3, 10, default_graph_config, default_scm_config, default_preprocessing_config, default_dataset_config, seed=42)

compare_dataloaders(loader_1, loader_2)